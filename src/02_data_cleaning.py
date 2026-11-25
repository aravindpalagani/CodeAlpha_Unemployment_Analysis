"""Phase 2: clean both unemployment datasets and document every decision.

Run from the project root:
    python src/02_data_cleaning.py

Outputs
-------
outputs/cleaned_data/unemployment_india_clean.csv
outputs/cleaned_data/unemployment_upto_11_2020_clean.csv
outputs/cleaned_data/cleaning_log.json
outputs/cleaned_data/data_quality_report.csv
outputs/cleaned_data/data_dictionary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils


def build_quality_checks(
    df1: pd.DataFrame, df2: pd.DataFrame, log1: dict, log2: dict
) -> pd.DataFrame:
    """Run independent post-cleaning checks and return a tidy report."""
    rows: list[dict] = []

    def add(dataset: str, check: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "dataset": dataset,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    d1_key = ["region", "date", "area"]
    d2_key = ["region", "date"]
    add(
        "Dataset 1",
        "No duplicate region-date-area keys",
        not df1.duplicated(d1_key).any(),
        f"{int(df1.duplicated(d1_key).sum())} duplicate keys",
    )
    add(
        "Dataset 2",
        "No duplicate region-date keys",
        not df2.duplicated(d2_key).any(),
        f"{int(df2.duplicated(d2_key).sum())} duplicate keys",
    )
    add(
        "Dataset 1",
        "No missing critical values",
        int(df1[utils.CRITICAL_COLUMNS_D1].isna().sum().sum()) == 0,
        f"{int(df1[utils.CRITICAL_COLUMNS_D1].isna().sum().sum())} missing critical cells",
    )
    add(
        "Dataset 2",
        "No missing critical values",
        int(df2[utils.CRITICAL_COLUMNS_D2].isna().sum().sum()) == 0,
        f"{int(df2[utils.CRITICAL_COLUMNS_D2].isna().sum().sum())} missing critical cells",
    )
    add(
        "Dataset 1",
        "Unemployment rate within 0-100",
        bool(df1["unemployment_rate"].between(0, 100).all()),
        f"range {df1['unemployment_rate'].min():.2f}-{df1['unemployment_rate'].max():.2f}",
    )
    add(
        "Dataset 2",
        "Unemployment rate within 0-100",
        bool(df2["unemployment_rate"].between(0, 100).all()),
        f"range {df2['unemployment_rate'].min():.2f}-{df2['unemployment_rate'].max():.2f}",
    )
    add(
        "Dataset 1",
        "Labour participation rate within 0-100",
        bool(df1["labour_participation_rate"].between(0, 100).all()),
        f"range {df1['labour_participation_rate'].min():.2f}-{df1['labour_participation_rate'].max():.2f}",
    )
    add(
        "Dataset 2",
        "Labour participation rate within 0-100",
        bool(df2["labour_participation_rate"].between(0, 100).all()),
        f"range {df2['labour_participation_rate'].min():.2f}-{df2['labour_participation_rate'].max():.2f}",
    )
    add(
        "Both",
        "Derived labour force is positive",
        bool((df1["estimated_labour_force"] > 0).all() and (df2["estimated_labour_force"] > 0).all()),
        "all derived labour-force estimates > 0",
    )
    add(
        "Dataset 1",
        "Dates parsed",
        bool(df1["date"].notna().all()),
        f"{df1['date'].min().date()} to {df1['date'].max().date()}",
    )
    add(
        "Dataset 2",
        "Dates parsed",
        bool(df2["date"].notna().all()),
        f"{df2['date'].min().date()} to {df2['date'].max().date()}",
    )
    add(
        "Dataset 1",
        "No completely empty rows remain",
        len(df1.dropna(how="all")) == len(df1),
        f"{len(df1)} rows retained",
    )
    add(
        "Dataset 2",
        "No completely empty rows remain",
        len(df2.dropna(how="all")) == len(df2),
        f"{len(df2)} rows retained",
    )
    add(
        "Dataset 1",
        "Missing unit-months left unimputed",
        True,
        (
            f"{log1['expected_region_area_months'] - len(df1)} region-area-month "
            "combinations are absent; they are not fabricated"
        ),
    )
    add(
        "Dataset 2",
        "Missing region-months left unimputed",
        True,
        (
            f"{log2['expected_region_months'] - len(df2)} region-month combinations "
            "are absent; they are not fabricated"
        ),
    )
    return pd.DataFrame(rows)


def add_coverage_metadata(
    df1: pd.DataFrame, df2: pd.DataFrame, log1: dict, log2: dict
) -> None:
    """Record expected panel size and missing unit-months in the cleaning logs."""
    log1["expected_region_area_months"] = int(
        df1["region"].nunique() * df1["area"].nunique() * df1["month"].nunique()
    )
    log1["observed_region_area_months"] = int(len(df1))
    log1["missing_region_area_months"] = int(
        log1["expected_region_area_months"] - len(df1)
    )
    log1["regions_with_incomplete_month_coverage"] = {
        region: int(n)
        for region, n in df1.groupby("region").size().items()
        if n < df1["month"].nunique() * df1["area"].nunique()
    }
    log2["expected_region_months"] = int(df2["region"].nunique() * df2["month"].nunique())
    log2["observed_region_months"] = int(len(df2))
    log2["missing_region_months"] = int(log2["expected_region_months"] - len(df2))
    expected = pd.MultiIndex.from_product(
        [sorted(df2["region"].unique()), sorted(df2["month"].unique())],
        names=["region", "month"],
    )
    observed = pd.MultiIndex.from_frame(df2[["region", "month"]])
    missing = expected.difference(observed)
    log2["missing_region_months_detail"] = [
        {"region": region, "month": month} for region, month in missing
    ]


def data_dictionary_markdown() -> str:
    """Return a Markdown data dictionary for both cleaned datasets."""
    return """# Cleaned Data Dictionary

Both cleaned files are stored in `outputs/cleaned_data/`. Column names use
`snake_case`; no values have been imputed.

## Common columns

| Column | Type | Description |
|---|---|---|
| `region` | text | Indian state or union territory as named in the source file. |
| `date` | date (YYYY-MM-DD) | Month-end observation date parsed from the source. |
| `frequency` | text | Standardised to `Monthly` (`Monthly` and `M` both mapped). |
| `unemployment_rate` | float | Estimated unemployment rate (% of the labour force). |
| `estimated_employed` | float | Estimated number of employed people. |
| `labour_participation_rate` | float | Estimated labour participation rate (%). |
| `month` | text (YYYY-MM) | Calendar month key for grouping. |
| `month_start` | date | First day of the month, used for time axes. |
| `year` | integer | Calendar year. |
| `month_name` | text | Three-letter month abbreviation. |
| `month_number` | integer | Calendar month number (1-12). |
| `period` | text | `Pre-COVID` (before Mar 2020), `Transition` (Mar 2020), or `COVID period` (Apr 2020 onward). |
| `estimated_labour_force` | float | Derived as `employed / (1 - unemployment_rate/100)`. |
| `estimated_unemployed` | float | Derived as `estimated_labour_force - estimated_employed`. |
| `zero_unemployment` | boolean | `True` where the source reports an exact 0% rate; retained and flagged, not removed. |

## Dataset 1 only

| Column | Type | Description |
|---|---|---|
| `area` | text | `Rural` or `Urban`. |

File: `unemployment_india_clean.csv` (740 rows, 28 regions, 14 months).

## Dataset 2 only

| Column | Type | Description |
|---|---|---|
| `zone` | text | Geographic grouping from the source (`North`, `South`, `East`, `West`, `Northeast`). |
| `longitude` | float | State centroid longitude supplied by the source. |
| `latitude` | float | State centroid latitude supplied by the source. |

File: `unemployment_upto_11_2020_clean.csv` (267 rows, 27 regions, 10 months).

## Important measurement note

`estimated_labour_force` and `estimated_unemployed` are arithmetic derivations,
not new official estimates. They are used to calculate labour-force-weighted
aggregates so that large states are not given the same weight as small ones.
The source does not provide population weights or survey design details.
"""


def validate(df1: pd.DataFrame, df2: pd.DataFrame, checks: pd.DataFrame) -> None:
    """Stop with a clear message if any post-cleaning integrity check fails."""
    failed = checks[checks["status"] != "PASS"]
    if not failed.empty:
        message = "\n".join(
            f"  - {row.dataset}: {row.check} ({row.detail})"
            for row in failed.itertuples()
        )
        raise ValueError(f"Data-quality checks failed:\n{message}")
    if df1.empty or df2.empty:
        raise ValueError("A cleaned dataset is empty; refusing to save results.")


def main() -> None:
    utils.ensure_directories()
    raw1 = utils.load_raw_dataset1()
    raw2 = utils.load_raw_dataset2()

    print("Cleaning Dataset 1...")
    clean1, log1 = utils.clean_dataset1(raw1)
    print("Cleaning Dataset 2...")
    clean2, log2 = utils.clean_dataset2(raw2)

    add_coverage_metadata(clean1, clean2, log1, log2)
    checks = build_quality_checks(clean1, clean2, log1, log2)
    validate(clean1, clean2, checks)

    utils.save_cleaned_dataset(clean1, utils.DATASET1_CLEAN)
    utils.save_cleaned_dataset(clean2, utils.DATASET2_CLEAN)

    cleaning_log = {
        "dataset1": log1,
        "dataset2": log2,
        "quality_checks": checks.to_dict(orient="records"),
        "not_merged_reason": (
            "Dataset 1 has Rural/Urban rows and 28 regions including Chandigarh; "
            "Dataset 2 has one overall row per region, 27 regions, geographic zones, "
            "and a longer 2020 window. They are related views, not duplicate records. "
            "They are analysed separately and compared only where definitions overlap."
        ),
    }
    utils.save_json(utils.CLEANED_DIR / "cleaning_log.json", cleaning_log)
    checks.to_csv(utils.CLEANED_DIR / "data_quality_report.csv", index=False)
    (utils.CLEANED_DIR / "data_dictionary.md").write_text(
        data_dictionary_markdown(), encoding="utf-8"
    )

    for label, log, frame in (
        ("Dataset 1", log1, clean1),
        ("Dataset 2", log2, clean2),
    ):
        print(f"\n{label}:")
        print(f"  raw rows: {log['raw_shape'][0]} -> cleaned rows: {frame.shape[0]}")
        print(f"  blank rows removed: {log['completely_blank_rows_removed']}")
        print(f"  duplicate rows removed: {log['duplicate_rows_removed']}")
        print(f"  text cells cleaned: {log['text_cells_cleaned']}")
        print(f"  date range: {log['date_range'][0]} to {log['date_range'][1]}")
        print(f"  zero unemployment rows retained: {log['zero_unemployment_rows_preserved']}")
        print(f"  regions: {log['regions']}; areas: {log.get('area_values', 'n/a')}")
        if "missing_region_area_months" in log:
            print(
                f"  missing region-area-months (not imputed): "
                f"{log['missing_region_area_months']}"
            )
        if "missing_region_months" in log:
            print(
                f"  missing region-months (not imputed): "
                f"{log['missing_region_months']} {log['missing_region_months_detail']}"
            )

    print(f"\nAll {len(checks)} quality checks passed.")
    print(f"Saved: {utils.DATASET1_CLEAN}")
    print(f"Saved: {utils.DATASET2_CLEAN}")
    print(f"Saved: {utils.CLEANED_DIR / 'cleaning_log.json'}")
    print(f"Saved: {utils.CLEANED_DIR / 'data_quality_report.csv'}")
    print(f"Saved: {utils.CLEANED_DIR / 'data_dictionary.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - surface a clear CLI error
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
