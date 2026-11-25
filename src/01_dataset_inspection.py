"""Phase 1: inspect both raw unemployment datasets without assumptions.

Run from the project root:
    python src/01_dataset_inspection.py

Outputs
-------
outputs/inspection/dataset_inspection.txt
outputs/inspection/inspection_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils


def _text_issues(series: pd.Series) -> int:
    """Count text cells with leading/trailing or repeated internal whitespace."""
    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return 0
    strings = series.dropna().astype("string")
    return int((strings != strings.str.replace(r"\s+", " ", regex=True).str.strip()).sum())


def inspect_dataset(
    raw: pd.DataFrame,
    dataset_name: str,
    date_column: str = "Date",
    expected_frequency_notes: str = "",
) -> tuple[list[str], dict]:
    """Build a text report and a machine-readable summary for one dataset."""
    lines: list[str] = []
    raw_columns = list(raw.columns)
    # Use a working copy with normalised headers for diagnostics, while still
    # reporting the original header strings (including whitespace).
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)
    lines.append("=" * 92)
    lines.append(dataset_name)
    lines.append("=" * 92)
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("Column names as stored in the CSV (repr highlights whitespace):")
    for column in raw_columns:
        lines.append(f"  - {column!r}")
    lines.append("")
    lines.append(f"Data types:\n{df.dtypes.to_string()}")
    lines.append("")
    lines.append(f"First five rows (headers normalised for display):\n{df.head(5).to_string(index=False)}")
    lines.append("")
    lines.append(f"Last five rows:\n{df.tail(5).to_string(index=False)}")
    lines.append("")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame(
        {"missing_count": missing, "missing_percent": missing_pct}
    )
    lines.append(f"Missing values by column:\n{missing_table.to_string()}")
    lines.append(f"Rows that are completely empty: {int(df.isna().all(axis=1).sum())}")
    lines.append(f"Exact duplicate rows (including empty rows): {int(df.duplicated().sum())}")
    non_empty = df.dropna(how="all")
    lines.append(
        "Exact duplicate rows after removing completely empty rows: "
        f"{int(non_empty.duplicated().sum())}"
    )
    lines.append("")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("Statistical summary of numeric columns:")
        lines.append(numeric.describe().T.round(2).to_string())
        lines.append("")

    # Date diagnosis (format is not assumed; both formats are attempted).
    date_series = df[date_column]
    parsed = pd.to_datetime(
        date_series.astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )
    parsed_alt = pd.to_datetime(
        date_series.astype("string").str.strip(), errors="coerce", dayfirst=True
    )
    invalid_primary = int(parsed[date_series.notna()].isna().sum())
    blank_dates = int(date_series.isna().sum())
    valid_dates = parsed.dropna()
    lines.append("Date diagnostics:")
    lines.append(f"  Raw example values: {date_series.dropna().head(3).tolist()}")
    lines.append(
        "  Leading/trailing whitespace cells in Date: "
        f"{_text_issues(date_series)}"
    )
    lines.append(f"  Blank date cells: {blank_dates}")
    lines.append(f"  Non-blank dates unparseable with dd-mm-yyyy: {invalid_primary}")
    lines.append(
        "  Fallback parse with dayfirst=True yields the same non-null count: "
        f"{int(parsed_alt.notna().sum()) == int(parsed.notna().sum())}"
    )
    if valid_dates.notna().any():
        lines.append(f"  Parsed range: {valid_dates.min().date()} to {valid_dates.max().date()}")
        month_counts = (
            valid_dates.dt.to_period("M").value_counts().sort_index().astype(int)
        )
        lines.append(f"  Distinct months: {month_counts.size}")
        lines.append("  Observations per month:")
        for month, count in month_counts.items():
            lines.append(f"    {month}: {count}")
    lines.append("")

    lines.append("Text-value diagnostics:")
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            values = series.dropna().astype("string").unique()
            lines.append(
                f"  {column!r}: {len(values)} unique values; "
                f"{_text_issues(series)} cells have irregular whitespace"
            )
            if len(values) <= 35:
                lines.append(f"    Values: {sorted(values.tolist())}")
    lines.append("")

    frequency = (
        df["Frequency"].astype("string").str.strip().value_counts(dropna=False).to_dict()
        if "Frequency" in df.columns
        else {}
    )
    area = (
        df["Area"].astype("string").str.strip().value_counts(dropna=False).to_dict()
        if "Area" in df.columns
        else {}
    )
    regions = (
        sorted(df["Region"].dropna().astype("string").str.strip().unique().tolist())
        if "Region" in df.columns
        else []
    )
    lines.append("Categorical overview:")
    lines.append(f"  Frequency values: {frequency}")
    if area:
        lines.append(f"  Area values: {area}")
    lines.append(f"  Number of regions: {len(regions)}")
    lines.append(f"  Regions: {regions}")
    if expected_frequency_notes:
        lines.append(f"  Note: {expected_frequency_notes}")
    lines.append("")

    summary = {
        "dataset": dataset_name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns_raw": [str(c) for c in raw_columns],
        "columns_have_leading_or_trailing_spaces": [
            c for c in raw_columns if c != c.strip()
        ],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
        "completely_empty_rows": int(df.isna().all(axis=1).sum()),
        "duplicate_rows_total": int(df.duplicated().sum()),
        "duplicate_rows_after_empty_removed": int(non_empty.duplicated().sum()),
        "date_range": [
            valid_dates.min().strftime("%Y-%m-%d"),
            valid_dates.max().strftime("%Y-%m-%d"),
        ]
        if len(valid_dates)
        else None,
        "distinct_months": int(valid_dates.dt.to_period("M").nunique()),
        "observations_per_month": {
            str(k): int(v)
            for k, v in valid_dates.dt.to_period("M").value_counts().sort_index().items()
        },
        "date_parse_failures": invalid_primary,
        "blank_date_cells": blank_dates,
        "region_count": len(regions),
        "regions": regions,
        "frequency_values": {str(k): int(v) for k, v in frequency.items()},
        "area_values": {str(k): int(v) for k, v in area.items()},
        "irregular_whitespace_cells": {
            column: _text_issues(df[column])
            for column in df.columns
            if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object
        },
        "numeric_summary": {
            str(column): {
                "count": int(numeric[column].count()),
                "mean": float(numeric[column].mean()),
                "std": float(numeric[column].std()),
                "min": float(numeric[column].min()),
                "p25": float(numeric[column].quantile(0.25)),
                "median": float(numeric[column].median()),
                "p75": float(numeric[column].quantile(0.75)),
                "max": float(numeric[column].max()),
            }
            for column in numeric.columns
        },
    }
    return lines, summary


def schema_comparison(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[list[str], dict]:
    """Document how the two datasets differ before any combination decision."""
    d1 = df1.copy()
    d2 = df2.copy()
    d1.columns = [str(c).strip() for c in d1.columns]
    d2.columns = [str(c).strip() for c in d2.columns]
    d1_dates = pd.to_datetime(d1["Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    d2_dates = pd.to_datetime(d2["Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    d1_regions = set(d1["Region"].dropna().str.strip())
    d2_regions = set(d2["Region"].dropna().str.strip())
    d1_months = set(d1_dates.dropna().dt.to_period("M").astype(str))
    d2_months = set(d2_dates.dropna().dt.to_period("M").astype(str))

    lines = [
        "=" * 92,
        "SCHEMA AND MEANING COMPARISON",
        "=" * 92,
        f"Common columns (after stripping names): {sorted(set(d1.columns) & set(d2.columns))}",
        f"Only in Dataset 1: {sorted(set(d1.columns) - set(d2.columns))}",
        f"Only in Dataset 2: {sorted(set(d2.columns) - set(d1.columns))}",
        f"Common regions: {len(d1_regions & d2_regions)}",
        f"Only in Dataset 1: {sorted(d1_regions - d2_regions)}",
        f"Only in Dataset 2: {sorted(d2_regions - d1_regions)}",
        f"Dataset 1 date range: {d1_dates.min().date()} to {d1_dates.max().date()}",
        f"Dataset 2 date range: {d2_dates.min().date()} to {d2_dates.max().date()}",
        f"Overlapping months: {sorted(d1_months & d2_months)}",
        "Dataset 1 has two rows per region-month: Rural and Urban.",
        "Dataset 2 has one row per region-month and no Rural/Urban column.",
        "Dataset 2 adds a geographic zone, longitude and latitude; Dataset 1 does not.",
        "The datasets therefore describe related but not identical populations. They",
        "must not be concatenated or duplicated into a single count series. Dataset 1",
        "is the primary source for area-level analysis; Dataset 2 is supplementary",
        "for the longer Jan-Oct 2020 recovery window and geographic zones.",
        "",
        "Dataset 2's filename says 'upto_11_2020', but the latest observation is",
        f"{d2_dates.max().date()}. The actual date range is used everywhere.",
        "",
    ]
    comparison = {
        "common_columns": sorted(set(d1.columns) & set(d2.columns)),
        "only_dataset1": sorted(set(d1.columns) - set(d2.columns)),
        "only_dataset2": sorted(set(d2.columns) - set(d1.columns)),
        "common_regions_count": len(d1_regions & d2_regions),
        "regions_only_dataset1": sorted(d1_regions - d2_regions),
        "regions_only_dataset2": sorted(d2_regions - d1_regions),
        "overlapping_months": sorted(d1_months & d2_months),
        "dataset1_date_range": [d1_dates.min().date().isoformat(), d1_dates.max().date().isoformat()],
        "dataset2_date_range": [d2_dates.min().date().isoformat(), d2_dates.max().date().isoformat()],
        "combination_decision": (
            "Do not concatenate. Dataset 1 supports Rural/Urban analysis; Dataset 2 "
            "supports a longer 2020 recovery window and geographic zones. Use Dataset 1 "
            "as primary and Dataset 2 as supplementary, with explicit schema notes."
        ),
    }
    return lines, comparison


def main() -> None:
    utils.ensure_directories()
    print("Inspecting raw datasets...\n")
    df1_raw = utils.load_raw_dataset1()
    df2_raw = utils.load_raw_dataset2()

    lines1, summary1 = inspect_dataset(
        df1_raw,
        "DATASET 1: Unemployment in India.csv (primary)",
        expected_frequency_notes="Frequency text is 'Monthly'; Area is Rural/Urban.",
    )
    lines2, summary2 = inspect_dataset(
        df2_raw,
        "DATASET 2: Unemployment_Rate_upto_11_2020.csv (supplementary)",
        expected_frequency_notes=(
            "Frequency is abbreviated 'M'; there is no Area column. The filename "
            "suggests November 2020, but the file ends on 31-10-2020."
        ),
    )
    lines3, comparison = schema_comparison(df1_raw, df2_raw)

    report_text = "\n".join(lines1 + lines2 + lines3)
    report_path = utils.INSPECTION_DIR / "dataset_inspection.txt"
    report_path.write_text(report_text, encoding="utf-8")

    payload = {
        "dataset1": summary1,
        "dataset2": summary2,
        "schema_comparison": comparison,
    }
    json_path = utils.INSPECTION_DIR / "inspection_summary.json"
    utils.save_json(json_path, payload)

    # Console summary (the complete report remains in the text file).
    for number, summary in (("1", summary1), ("2", summary2)):
        print(f"Dataset {number}: {summary['shape']['rows']} rows x {summary['shape']['columns']} columns")
        print(f"  columns: {summary['columns_raw']}")
        print(f"  date range: {summary['date_range']} ({summary['distinct_months']} months)")
        print(f"  empty rows: {summary['completely_empty_rows']}")
        print(f"  duplicate rows (raw / after empty removal): "
              f"{summary['duplicate_rows_total']} / {summary['duplicate_rows_after_empty_removed']}")
        print(f"  total missing cells: {sum(summary['missing_by_column'].values())}")
        print(f"  frequency values: {summary['frequency_values']}")
        if summary["area_values"]:
            print(f"  area values: {summary['area_values']}")
        print()

    print("Schema comparison:")
    print(f"  common regions: {comparison['common_regions_count']}")
    print(f"  regions only in Dataset 1: {comparison['regions_only_dataset1']}")
    print(f"  overlapping months: {comparison['overlapping_months']}")
    print(f"  decision: {comparison['combination_decision']}")
    print(f"\nSaved: {report_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - surface a clear CLI error
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
