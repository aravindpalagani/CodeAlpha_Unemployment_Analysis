"""Shared utilities for the CodeAlpha Unemployment Analysis project.

The module centralises project paths, data-cleaning logic, derived labour-market
measures and the matplotlib/seaborn theme used by every script and the notebook.
All paths are resolved relative to the project root so the project can be moved
to another machine without editing code.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # script-friendly backend; notebooks keep their inline backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NOTEBOOK_DIR = ROOT / "notebooks"
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
CLEANED_DIR = OUTPUT_DIR / "cleaned_data"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
INSPECTION_DIR = OUTPUT_DIR / "inspection"
REPORT_DIR = ROOT / "report"

DATASET1_RAW = DATA_DIR / "Unemployment in India.csv"
DATASET2_RAW = DATA_DIR / "Unemployment_Rate_upto_11_2020.csv"
DATASET1_CLEAN = CLEANED_DIR / "unemployment_india_clean.csv"
DATASET2_CLEAN = CLEANED_DIR / "unemployment_upto_11_2020_clean.csv"

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------
COLORS = {
    "navy": "#17365D",
    "blue": "#2E6DA4",
    "teal": "#2A9D8F",
    "orange": "#E08A2E",
    "red": "#C0392B",
    "gold": "#D4A017",
    "gray": "#6B7280",
    "light_gray": "#D9DEE5",
    "green": "#4C956C",
    "purple": "#7B5EA7",
}
AREA_COLORS = {"Rural": COLORS["green"], "Urban": COLORS["blue"]}
PERIOD_COLORS = {
    "Pre-COVID": COLORS["blue"],
    "Transition": COLORS["gold"],
    "COVID period": COLORS["red"],
}
COVID_START = pd.Timestamp("2020-03-01")
COVID_SHOCK_START = pd.Timestamp("2020-04-01")

# Clean column names used throughout the analysis.
DATASET1_RENAME = {
    "Region": "region",
    "Date": "date_raw",
    "Frequency": "frequency",
    "Estimated Unemployment Rate (%)": "unemployment_rate",
    "Estimated Employed": "estimated_employed",
    "Estimated Labour Participation Rate (%)": "labour_participation_rate",
    "Area": "area",
}
DATASET2_RENAME = {
    "Region": "region",
    "Date": "date_raw",
    "Frequency": "frequency",
    "Estimated Unemployment Rate (%)": "unemployment_rate",
    "Estimated Employed": "estimated_employed",
    "Estimated Labour Participation Rate (%)": "labour_participation_rate",
    "Region.1": "zone",
    "longitude": "longitude",
    "latitude": "latitude",
}
NUMERIC_COLUMNS = [
    "unemployment_rate",
    "estimated_employed",
    "labour_participation_rate",
]
TEXT_COLUMNS_D1 = ["Region", "Date", "Frequency", "Area"]
TEXT_COLUMNS_D2 = ["Region", "Date", "Frequency", "Region.1"]
CRITICAL_COLUMNS_D1 = [
    "region",
    "date",
    "frequency",
    "unemployment_rate",
    "estimated_employed",
    "labour_participation_rate",
    "area",
]
CRITICAL_COLUMNS_D2 = [
    "region",
    "date",
    "frequency",
    "unemployment_rate",
    "estimated_employed",
    "labour_participation_rate",
    "zone",
]


def ensure_directories() -> None:
    """Create every directory the project writes to."""
    for directory in (
        DATA_DIR,
        NOTEBOOK_DIR,
        SRC_DIR,
        OUTPUT_DIR,
        CHART_DIR,
        CLEANED_DIR,
        ANALYSIS_DIR,
        INSPECTION_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """Apply a consistent, readable theme to all charts."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlecolor": COLORS["navy"],
            "axes.labelsize": 11,
            "axes.labelcolor": "#263238",
            "axes.edgecolor": "#7A8794",
            "axes.linewidth": 0.9,
            "xtick.color": "#37474F",
            "ytick.color": "#37474F",
            "grid.color": "#DDE3EA",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_csv_checked(path: Path) -> pd.DataFrame:
    """Read a CSV and raise a clear error when the expected file is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the original CSV files in the project's data/ directory."
        )
    return pd.read_csv(path)


def load_raw_dataset1() -> pd.DataFrame:
    return read_csv_checked(DATASET1_RAW)


def load_raw_dataset2() -> pd.DataFrame:
    return read_csv_checked(DATASET2_RAW)


def load_cleaned() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both cleaned datasets, parsing the date fields explicitly."""
    if not DATASET1_CLEAN.exists() or not DATASET2_CLEAN.exists():
        raise FileNotFoundError(
            "Cleaned datasets are missing. Run src/02_data_cleaning.py first."
        )
    df1 = pd.read_csv(
        DATASET1_CLEAN,
        parse_dates=["date", "month_start"],
    )
    df2 = pd.read_csv(
        DATASET2_CLEAN,
        parse_dates=["date", "month_start"],
    )
    return df1, df2


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def _standardise_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Strip whitespace and BOM characters from column names."""
    original = list(df.columns)
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    changed = sum(1 for old, new in zip(original, df.columns) if old != new)
    return df, changed


def _clean_text_cells(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Collapse internal whitespace and strip leading/trailing spaces in text."""
    df = df.copy()
    changed = 0
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            cleaned = (
                series.astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            changed += int((cleaned.fillna("") != series.fillna("").astype("string")).sum())
            df[column] = cleaned
    return df, changed


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, period and calendar helper columns used across the project."""
    df = df.copy()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["month_start"] = df["date"].values.astype("datetime64[M]")
    df["year"] = df["date"].dt.year
    df["month_name"] = df["date"].dt.strftime("%b")
    df["month_number"] = df["date"].dt.month
    df["period"] = np.select(
        [
            df["date"] < COVID_START,
            df["date"] < COVID_SHOCK_START,
        ],
        ["Pre-COVID", "Transition"],
        default="COVID period",
    )
    return df


def add_labour_force_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Derive estimated labour force and unemployed counts.

    ``Estimated Employed`` is a count and the unemployment rate is the share of
    the labour force without work. Therefore:
        labour force = employed / (1 - unemployment_rate / 100)
        unemployed   = labour force - employed
    This lets us calculate a labour-force-weighted aggregate rather than taking
    a simple average of rates across regions of very different sizes.
    """
    df = df.copy()
    rate = df["unemployment_rate"].astype(float)
    employed = df["estimated_employed"].astype(float)
    valid = rate.ge(0) & rate.lt(100) & employed.gt(0)
    labour_force = pd.Series(np.nan, index=df.index, dtype=float)
    labour_force.loc[valid] = employed.loc[valid] / (1 - rate.loc[valid] / 100)
    df["estimated_labour_force"] = labour_force
    df["estimated_unemployed"] = labour_force - employed
    return df


def _clean_dataset_common(
    raw: pd.DataFrame,
    dataset_name: str,
    rename_map: dict[str, str],
    text_columns: list[str],
    critical_columns: list[str],
    dataset_number: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shared cleaning workflow for both unemployment datasets."""
    log: dict[str, Any] = {
        "dataset": dataset_name,
        "raw_shape": list(raw.shape),
        "raw_columns": [repr(c) for c in raw.columns],
    }
    df = raw.copy()
    df, header_changes = _standardise_headers(df)
    log["column_names_standardised"] = header_changes

    blank_rows = int(df.isna().all(axis=1).sum())
    df = df.dropna(how="all").copy()
    log["completely_blank_rows_removed"] = blank_rows

    df, text_cells_changed = _clean_text_cells(df, text_columns)
    log["text_cells_cleaned"] = text_cells_changed

    raw_duplicates = int(raw.duplicated().sum())
    duplicates_after_blank_drop = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    log["duplicate_rows_including_blank"] = raw_duplicates
    log["duplicate_rows_after_blank_drop"] = duplicates_after_blank_drop
    log["duplicate_rows_removed"] = duplicates_after_blank_drop

    renamed = df.rename(columns=rename_map)
    log["renamed_columns"] = {old: new for old, new in rename_map.items()}

    # Dates are stored as day-month-year strings with leading whitespace.
    parsed_dates = pd.to_datetime(
        renamed["date_raw"].astype("string").str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )
    invalid_dates = int(parsed_dates.isna().sum())
    renamed["date"] = parsed_dates
    renamed = renamed.drop(columns=["date_raw"])
    log["invalid_dates"] = invalid_dates

    for column in NUMERIC_COLUMNS:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    log["numeric_columns_coerced"] = [
        col for col in NUMERIC_COLUMNS if renamed[col].isna().any()
    ]

    renamed["frequency"] = renamed["frequency"].replace(
        {"Monthly": "Monthly", "M": "Monthly", "Monthly ": "Monthly"}
    )

    missing_before = int(renamed[critical_columns].isna().any(axis=1).sum())
    renamed = renamed.dropna(subset=critical_columns).copy()
    log["rows_with_missing_critical_values_removed"] = missing_before
    log["remaining_missing_cells"] = int(renamed[critical_columns].isna().sum().sum())

    # Definitional range checks. Unusual values are preserved but recorded.
    rate = renamed["unemployment_rate"]
    lpr = renamed["labour_participation_rate"]
    employed = renamed["estimated_employed"]
    log["invalid_rate_outside_0_100"] = int((~rate.between(0, 100)).sum())
    log["invalid_lpr_outside_0_100"] = int((~lpr.between(0, 100)).sum())
    log["invalid_nonpositive_employment"] = int((employed <= 0).sum())
    log["zero_unemployment_rows_preserved"] = int(rate.eq(0).sum())
    log["low_lpr_below_20_rows"] = int((lpr < 20).sum())
    log["numerical_ranges"] = {
        "unemployment_rate": [float(rate.min()), float(rate.max())],
        "estimated_employed": [float(employed.min()), float(employed.max())],
        "labour_participation_rate": [float(lpr.min()), float(lpr.max())],
    }

    renamed = _add_time_features(renamed)
    renamed = add_labour_force_estimates(renamed)
    renamed["zero_unemployment"] = renamed["unemployment_rate"].eq(0)
    renamed = renamed.sort_values(
        ["region", "area", "date"] if "area" in renamed.columns else ["region", "date"]
    ).reset_index(drop=True)

    ordered = [
        "region",
        "date",
        "frequency",
        "unemployment_rate",
        "estimated_employed",
        "labour_participation_rate",
    ]
    if "area" in renamed.columns:
        ordered.append("area")
    if "zone" in renamed.columns:
        ordered.extend(["zone", "longitude", "latitude"])
    ordered.extend(
        [
            "month",
            "month_start",
            "year",
            "month_name",
            "month_number",
            "period",
            "estimated_labour_force",
            "estimated_unemployed",
            "zero_unemployment",
        ]
    )
    renamed = renamed[ordered]

    log["cleaned_shape"] = list(renamed.shape)
    log["date_range"] = [
        renamed["date"].min().strftime("%Y-%m-%d"),
        renamed["date"].max().strftime("%Y-%m-%d"),
    ]
    log["regions"] = int(renamed["region"].nunique())
    log["frequency_values"] = sorted(renamed["frequency"].unique().tolist())
    if "area" in renamed.columns:
        log["area_values"] = sorted(renamed["area"].unique().tolist())
    log["missing_value_policy"] = (
        "No missing values were imputed. Completely empty rows were removed; "
        "states with unreported region-months remain absent from those months. "
        "Definitional zero unemployment values were retained and flagged."
    )
    return renamed, log


def clean_dataset1(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean Dataset 1: regional data with a Rural/Urban split."""
    return _clean_dataset_common(
        raw=raw,
        dataset_name="Unemployment in India (primary dataset)",
        rename_map=DATASET1_RENAME,
        text_columns=TEXT_COLUMNS_D1,
        critical_columns=CRITICAL_COLUMNS_D1,
        dataset_number=1,
    )


def clean_dataset2(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean Dataset 2: regional data with geographic zone and coordinates."""
    return _clean_dataset_common(
        raw=raw,
        dataset_name="Unemployment Rate up to 11/2020 (supplementary dataset)",
        rename_map=DATASET2_RENAME,
        text_columns=TEXT_COLUMNS_D2,
        critical_columns=CRITICAL_COLUMNS_D2,
        dataset_number=2,
    )


def save_cleaned_dataset(df: pd.DataFrame, path: Path) -> None:
    """Write a cleaned CSV with consistent date formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")


# ---------------------------------------------------------------------------
# Aggregation and serialisation helpers
# ---------------------------------------------------------------------------
def weighted_rate(df: pd.DataFrame) -> float:
    """Return the labour-force-weighted unemployment rate for a subset."""
    if df.empty:
        return float("nan")
    if "estimated_labour_force" not in df.columns:
        df = add_labour_force_estimates(df)
    labour_force = float(df["estimated_labour_force"].sum())
    unemployed = float(df["estimated_unemployed"].sum())
    return float(unemployed / labour_force * 100) if labour_force > 0 else float("nan")


def weighted_lpr(df: pd.DataFrame) -> float:
    """Approximate aggregate labour participation rate.

    LPR = labour force / population, so population is imputed as LF / (LPR/100)
    for each observation. The aggregate is then total LF / total implied
    population. This is an approximation because population is not supplied.
    """
    if df.empty:
        return float("nan")
    if "estimated_labour_force" not in df.columns:
        df = add_labour_force_estimates(df)
    valid = df["labour_participation_rate"].gt(0)
    if not valid.any():
        return float("nan")
    labour_force = float(df.loc[valid, "estimated_labour_force"].sum())
    population = float(
        (df.loc[valid, "estimated_labour_force"] / (df.loc[valid, "labour_participation_rate"] / 100)).sum()
    )
    return float(labour_force / population * 100) if population > 0 else float("nan")


def save_json(path: Path, data: Any) -> None:
    """Serialise nested analysis results, including numpy/pandas scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (float,)):
        return None if np.isnan(value) else value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to outputs/charts at publication quality."""
    ensure_directories()
    path = CHART_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return path
