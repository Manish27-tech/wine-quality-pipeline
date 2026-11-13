"""Dataset acquisition, loading and schema validation."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when a raw file does not satisfy the expected schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Download the red and white wine CSVs, caching them on disk."""
    paths: dict[str, Path] = {}
    for wine_type, filename in config.RAW_FILES.items():
        destination = config.RAW_DIR / filename
        if destination.exists() and not force:
            paths[wine_type] = destination
            if verbose:
                print(f"[data] cached {filename} ({destination.stat().st_size / 1024:.0f} KB)")
            continue

        url = f"{config.BASE_URL}/{filename}"
        try:
            if verbose:
                print(f"[data] downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if len(payload) < 50_000:
                raise ValueError(f"payload too small ({len(payload)} bytes)")
            destination.write_bytes(payload)
            if verbose:
                print(f"[data] saved {destination.name} "
                      f"(sha256={_sha256(destination)[:12]}...)")
            paths[wine_type] = destination
        except Exception as exc:  # noqa: BLE001 - surfaced as a single error
            raise RuntimeError(f"Could not download {url}: {exc}") from exc
    return paths


def load_raw(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    """Load both CSVs, tag the grape type and concatenate.

    UCI ships both files semicolon-separated with quoted headers.
    """
    paths = paths or download(verbose=False)
    frames = []
    for wine_type, path in paths.items():
        frame = pd.read_csv(path, sep=";")
        frame.columns = [column.strip() for column in frame.columns]
        frame[config.WINE_TYPE] = wine_type
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined[config.WINE_TYPE] = combined[config.WINE_TYPE].astype("category")
    return combined


def validate(df: pd.DataFrame) -> dict:
    """Check schema, ranges and row counts; raise on critical failures."""
    checks: list[dict] = []
    critical_failures: list[str] = []

    def record(name: str, passed: bool, detail: str, critical: bool = False) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail,
                       "critical": critical})
        if not passed and critical:
            critical_failures.append(name)

    missing_columns = [c for c in [*config.BASE_FEATURES, config.TARGET] if c not in df.columns]
    record("expected_columns", not missing_columns,
           "all 11 physicochemical columns present" if not missing_columns
           else f"missing: {missing_columns}", critical=True)
    if critical_failures:
        raise DataValidationError(
            "Dataset failed critical validation: " + ", ".join(critical_failures)
        )

    record("non_empty", len(df) > 0, f"{len(df)} rows", critical=True)
    record("expected_scale", len(df) >= 6000,
           f"{len(df)} rows (full dataset has ~6,500)")

    types = set(df[config.WINE_TYPE].astype(str).unique())
    record("wine_types", types == {"red", "white"}, f"observed: {sorted(types)}", critical=True)

    null_counts = {column: int(df[column].isna().sum()) for column in config.BASE_FEATURES}
    total_nulls = sum(null_counts.values())
    record("no_missing_measurements", total_nulls == 0, f"{total_nulls} nulls in features",
           critical=True)

    out_of_range = {}
    for column, (low, high) in config.VALID_RANGES.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        count = int((~series.between(low, high)).sum())
        if count:
            out_of_range[column] = count
    record("physical_ranges", not out_of_range,
           "all measurements within plausible bounds" if not out_of_range
           else f"out of range: {out_of_range}")

    quality_domain = sorted(map(int, df[config.TARGET].dropna().unique()))
    record("quality_domain", set(quality_domain) <= set(range(0, 11)),
           f"observed: {quality_domain}")

    duplicates = int(df.duplicated().sum())
    record("duplicate_rows_reported", True,
           f"{duplicates} exact duplicates (dropped during cleaning, not a failure)")

    if critical_failures:
        raise DataValidationError(
            "Dataset failed critical validation: " + ", ".join(critical_failures)
        )

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "rows_by_type": {k: int(v) for k, v in
                         df[config.WINE_TYPE].astype(str).value_counts().items()},
        "duplicate_rows": duplicates,
        "checks": checks,
    }
    df.attrs["validation"] = report
    return report
