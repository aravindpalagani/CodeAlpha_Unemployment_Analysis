"""Phase 3: exploratory data analysis with numerical summaries.

Run from the project root (after cleaning):
    python src/03_exploratory_analysis.py

Outputs
-------
outputs/analysis/eda_summary.json
outputs/analysis/state_summary.csv
outputs/analysis/area_summary.csv
outputs/analysis/monthly_summary.csv
outputs/analysis/area_monthly_summary.csv
outputs/analysis/correlation_pearson.csv
outputs/analysis/correlation_spearman.csv
outputs/analysis/calendar_month_summary.csv
outputs/analysis/balanced_panel_monthly.csv
outputs/analysis/coverage_by_region.csv
outputs/analysis/outliers.csv
outputs/analysis/highest_observations.csv
outputs/analysis/zero_unemployment_observations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly national aggregates, including weighted and unweighted rates."""
    records: list[dict] = []
    for month, group in df.groupby("month_start", sort=True):
        records.append(
            {
                "month": group["month"].iloc[0],
                "month_start": month,
                "n_obs": len(group),
                "n_regions": group["region"].nunique(),
                "n_rural_obs": int((group["area"] == "Rural").sum()),
                "n_urban_obs": int((group["area"] == "Urban").sum()),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "weighted_rate": utils.weighted_rate(group),
                "median_rate": group["unemployment_rate"].median(),
                "p25_rate": group["unemployment_rate"].quantile(0.25),
                "p75_rate": group["unemployment_rate"].quantile(0.75),
                "min_rate": group["unemployment_rate"].min(),
                "max_rate": group["unemployment_rate"].max(),
                "avg_lpr": group["labour_participation_rate"].mean(),
                "weighted_lpr": utils.weighted_lpr(group),
                "total_employed_reported": group["estimated_employed"].sum(),
                "total_labour_force": group["estimated_labour_force"].sum(),
                "total_unemployed": group["estimated_unemployed"].sum(),
            }
        )
    return pd.DataFrame(records)


def state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Region-level summary with weighted rates and coverage information."""
    grouped = df.assign(
        _unemployed=df["estimated_unemployed"],
        _labour_force=df["estimated_labour_force"],
    ).groupby("region", observed=True)
    summary = grouped.agg(
        n_obs=("unemployment_rate", "size"),
        months_observed=("month", "nunique"),
        mean_rate=("unemployment_rate", "mean"),
        median_rate=("unemployment_rate", "median"),
        min_rate=("unemployment_rate", "min"),
        max_rate=("unemployment_rate", "max"),
        avg_lpr=("labour_participation_rate", "mean"),
        avg_employed=("estimated_employed", "mean"),
        total_unemployed=("_unemployed", "sum"),
        total_labour_force=("_labour_force", "sum"),
    )
    summary["weighted_rate"] = (
        summary["total_unemployed"] / summary["total_labour_force"] * 100
    )
    summary = summary.drop(columns=["total_unemployed", "total_labour_force"])
    rural = (
        df[df["area"] == "Rural"].groupby("region").size().rename("n_rural_obs")
    )
    urban = (
        df[df["area"] == "Urban"].groupby("region").size().rename("n_urban_obs")
    )
    summary = summary.join(rural).join(urban).fillna({"n_rural_obs": 0, "n_urban_obs": 0})
    expected = df["month"].nunique() * df["area"].nunique()
    summary["coverage_complete"] = summary["n_obs"].eq(expected)
    summary = summary.reset_index().sort_values("weighted_rate", ascending=False)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def area_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rural/urban summary with weighted and unweighted means."""
    records = []
    for area, group in df.groupby("area", observed=True):
        records.append(
            {
                "area": area,
                "n_obs": len(group),
                "n_regions": group["region"].nunique(),
                "months": group["month"].nunique(),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "weighted_rate": utils.weighted_rate(group),
                "median_rate": group["unemployment_rate"].median(),
                "min_rate": group["unemployment_rate"].min(),
                "max_rate": group["unemployment_rate"].max(),
                "avg_lpr": group["labour_participation_rate"].mean(),
                "avg_employed": group["estimated_employed"].mean(),
                "total_employed": group["estimated_employed"].sum(),
            }
        )
    return pd.DataFrame(records).sort_values("weighted_rate", ascending=False)


def area_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly weighted rates and LPR separately for Rural and Urban."""
    records = []
    for (month_start, area), group in df.groupby(["month_start", "area"]):
        records.append(
            {
                "month": group["month"].iloc[0],
                "month_start": month_start,
                "area": area,
                "n_obs": len(group),
                "weighted_rate": utils.weighted_rate(group),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "avg_lpr": group["labour_participation_rate"].mean(),
                "total_employed_reported": group["estimated_employed"].sum(),
            }
        )
    return pd.DataFrame(records).sort_values(["month_start", "area"])


def calendar_month_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month averages and year coverage for a seasonality check."""
    records = []
    for month_number, group in df.groupby("month_number", observed=True):
        records.append(
            {
                "month_number": int(month_number),
                "month_name": group["month_name"].iloc[0],
                "years_covered": sorted(group["year"].unique().tolist()),
                "n_years": group["year"].nunique(),
                "n_obs": len(group),
                "weighted_rate": utils.weighted_rate(group),
                "unweighted_rate": group["unemployment_rate"].mean(),
            }
        )
    return pd.DataFrame(records).sort_values("month_number")


def balanced_panel_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compare the full sample with a balanced panel of completely covered states."""
    months = df["month"].nunique()
    areas = df["area"].nunique()
    counts = df.groupby("region").size()
    complete_states = counts[counts == months * areas].index.tolist()
    panel = df[df["region"].isin(complete_states)]
    full = monthly_summary(df).set_index("month")
    balanced_records = []
    for month, group in panel.groupby("month_start", sort=True):
        month_key = group["month"].iloc[0]
        balanced_records.append(
            {
                "month": month_key,
                "month_start": month,
                "n_obs": len(group),
                "weighted_rate_balanced": utils.weighted_rate(group),
                "unweighted_rate_balanced": group["unemployment_rate"].mean(),
                "total_employed_balanced": group["estimated_employed"].sum(),
                "weighted_rate_full": full.loc[month_key, "weighted_rate"],
                "difference_pp": utils.weighted_rate(group) - full.loc[month_key, "weighted_rate"],
            }
        )
    return pd.DataFrame(balanced_records), complete_states


def build_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """IQR-based outlier check; flagged observations are retained."""
    q1 = float(df["unemployment_rate"].quantile(0.25))
    q3 = float(df["unemployment_rate"].quantile(0.75))
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    outliers = df[
        (df["unemployment_rate"] < lower) | (df["unemployment_rate"] > upper)
    ].copy()
    outliers["outlier_direction"] = np.where(
        outliers["unemployment_rate"] > upper, "high", "low"
    )
    outliers = outliers.sort_values("unemployment_rate", ascending=False)
    summary = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outlier_count": int(len(outliers)),
        "outlier_share_pct": float(len(outliers) / len(df) * 100),
        "high_outlier_count": int((outliers["outlier_direction"] == "high").sum()),
        "low_outlier_count": int((outliers["outlier_direction"] == "low").sum()),
        "treatment": (
            "Retained. These are plausible extreme labour-market observations, "
            "especially during April-May 2020; removing them would understate the shock."
        ),
    }
    return outliers, summary


def relationships(df: pd.DataFrame, monthly: pd.DataFrame, state: pd.DataFrame) -> dict:
    """Compute the relationships reported in the EDA narrative."""
    raw_cols = [
        "unemployment_rate",
        "estimated_employed",
        "labour_participation_rate",
    ]
    pearson = df[raw_cols].corr(method="pearson")
    spearman = df[raw_cols].corr(method="spearman")
    state_corr = state[["mean_rate", "avg_lpr"]].corr().iloc[0, 1]
    monthly_employment_corr = monthly["unweighted_rate"].corr(
        monthly["total_employed_reported"]
    )
    return {
        "pearson": pearson.round(4).to_dict(),
        "spearman": spearman.round(4).to_dict(),
        "state_average_unemployment_vs_lpr_pearson": float(state_corr),
        "monthly_unweighted_unemployment_vs_reported_employment_pearson": float(
            monthly_employment_corr
        ),
        "interpretation": (
            "Pooled correlations are weak because state size and repeated observations "
            "dominate the cross-section. The monthly national association between "
            "unemployment and total reported employment is strongly negative, but both "
            "series are mechanically linked to the labour force and neither establishes "
            "causality."
        ),
    }


def seasonality_assessment(df: pd.DataFrame) -> dict:
    """State explicitly whether the data can support a seasonality claim."""
    years = sorted(df["year"].unique().tolist())
    two_year_months = (
        df.groupby("month_number")["year"].nunique().loc[lambda s: s >= 2].index.tolist()
    )
    repeated = (
        df[df["month_number"].isin(two_year_months)]
        .groupby(["month_number", "month_name", "year"], observed=True)["unemployment_rate"]
        .mean()
        .reset_index()
    )
    return {
        "assessment": "insufficient_evidence",
        "years_covered": years,
        "months_covered": int(df["month"].nunique()),
        "calendar_months_observed_in_both_years": two_year_months,
        "note": (
            "The primary dataset spans only 14 consecutive months (May 2019-June 2020). "
            "Only May and June appear in both years, and the 2020 repeats coincide with "
            "the COVID-19 shock. A reliable seasonal cycle normally requires multiple "
            "complete years. No seasonal effect is claimed. The supplementary dataset "
            "covers a single calendar year (Jan-Oct 2020)."
        ),
        "repeated_month_values": repeated.round(2).to_dict(orient="records"),
    }


def main() -> None:
    utils.ensure_directories()
    df1, df2 = utils.load_cleaned()

    monthly = monthly_summary(df1)
    state = state_summary(df1)
    area = area_summary(df1)
    area_monthly = area_monthly_summary(df1)
    calendar = calendar_month_summary(df1)
    balanced, complete_states = balanced_panel_summary(df1)
    outliers, outlier_summary = build_outliers(df1)
    rel = relationships(df1, monthly, state)
    season = seasonality_assessment(df1)

    pearson = df1[
        ["unemployment_rate", "estimated_employed", "labour_participation_rate"]
    ].corr(method="pearson")
    spearman = df1[
        ["unemployment_rate", "estimated_employed", "labour_participation_rate"]
    ].corr(method="spearman")

    peak = monthly.loc[monthly["weighted_rate"].idxmax()]
    trough_states = state.nsmallest(5, "weighted_rate")
    top_states = state.nlargest(5, "weighted_rate")
    overall_weighted = utils.weighted_rate(df1)
    overall = {
        "observations": int(len(df1)),
        "regions": int(df1["region"].nunique()),
        "areas": sorted(df1["area"].unique().tolist()),
        "months": int(df1["month"].nunique()),
        "date_range": [
            df1["date"].min().strftime("%Y-%m-%d"),
            df1["date"].max().strftime("%Y-%m-%d"),
        ],
        "weighted_rate": overall_weighted,
        "unweighted_mean": float(df1["unemployment_rate"].mean()),
        "median": float(df1["unemployment_rate"].median()),
        "std": float(df1["unemployment_rate"].std()),
        "min": float(df1["unemployment_rate"].min()),
        "max": float(df1["unemployment_rate"].max()),
        "p25": float(df1["unemployment_rate"].quantile(0.25)),
        "p75": float(df1["unemployment_rate"].quantile(0.75)),
        "zero_unemployment_observations": int(df1["zero_unemployment"].sum()),
        "peak_month": {
            "month": peak["month"],
            "weighted_rate": float(peak["weighted_rate"]),
            "unweighted_rate": float(peak["unweighted_rate"]),
            "n_obs": int(peak["n_obs"]),
        },
        "estimated_employed_mean_per_observation": float(
            df1["estimated_employed"].mean()
        ),
        "average_lpr": float(df1["labour_participation_rate"].mean()),
    }

    coverage = {
        "dataset1": {
            "rows": int(len(df1)),
            "regions": int(df1["region"].nunique()),
            "months": int(df1["month"].nunique()),
            "area_units": int(df1["area"].nunique()),
            "expected_region_area_months": int(
                df1["region"].nunique() * df1["area"].nunique() * df1["month"].nunique()
            ),
            "missing_region_area_months": int(
                df1["region"].nunique() * df1["area"].nunique() * df1["month"].nunique()
                - len(df1)
            ),
            "complete_coverage_states": len(complete_states),
            "complete_state_names": complete_states,
            "incomplete_coverage_states": sorted(
                set(df1["region"].unique()) - set(complete_states)
            ),
        },
        "dataset2": {
            "rows": int(len(df2)),
            "regions": int(df2["region"].nunique()),
            "months": int(df2["month"].nunique()),
            "date_range": [
                df2["date"].min().strftime("%Y-%m-%d"),
                df2["date"].max().strftime("%Y-%m-%d"),
            ],
            "zones": sorted(df2["zone"].unique().tolist()),
        },
        "balanced_panel_check": {
            "max_abs_weighted_difference_pp": float(balanced["difference_pp"].abs().max()),
            "mean_abs_weighted_difference_pp": float(balanced["difference_pp"].abs().mean()),
            "interpretation": (
                "Recomputing the monthly trend on only the 20 states with all 14 months "
                "and both areas produces nearly identical weighted rates. The national "
                "trend is therefore not an artefact of changing state coverage."
            ),
        },
    }

    top_bottom = {
        "highest": top_states[
            ["region", "weighted_rate", "mean_rate", "n_obs", "months_observed"]
        ].to_dict(orient="records"),
        "lowest": trough_states[
            ["region", "weighted_rate", "mean_rate", "n_obs", "months_observed"]
        ].to_dict(orient="records"),
    }

    area_records = area.round(4).to_dict(orient="records")
    monthly_records = monthly.copy()
    monthly_records["month_start"] = monthly_records["month_start"].dt.strftime("%Y-%m-%d")
    monthly_records = monthly_records.round(4).to_dict(orient="records")

    summary = {
        "dataset1_overall": overall,
        "dataset2_overview": coverage["dataset2"],
        "area_summary": area_records,
        "top_bottom_states": top_bottom,
        "coverage": coverage,
        "relationships": rel,
        "outliers": outlier_summary,
        "seasonality": season,
        "method_notes": {
            "weighted_rate": (
                "Labour-force weighted: sum(estimated unemployed) / sum(estimated labour "
                "force) x 100, with labour force derived from employed/(1-rate/100)."
            ),
            "unweighted_mean": (
                "Simple mean of all available region-area-month unemployment rates; "
                "each reporting unit has equal weight."
            ),
            "period_definitions": {
                "Pre-COVID": "Observations before 2020-03-01",
                "Transition": "March 2020 (lockdown began 25 March)",
                "COVID period": "April 2020 onward",
            },
        },
    }

    # Persist tables.
    monthly.to_csv(utils.ANALYSIS_DIR / "monthly_summary.csv", index=False, date_format="%Y-%m-%d")
    state.to_csv(utils.ANALYSIS_DIR / "state_summary.csv", index=False)
    area.to_csv(utils.ANALYSIS_DIR / "area_summary.csv", index=False)
    area_monthly.to_csv(
        utils.ANALYSIS_DIR / "area_monthly_summary.csv", index=False, date_format="%Y-%m-%d"
    )
    pearson.round(4).to_csv(utils.ANALYSIS_DIR / "correlation_pearson.csv")
    spearman.round(4).to_csv(utils.ANALYSIS_DIR / "correlation_spearman.csv")
    calendar.to_csv(utils.ANALYSIS_DIR / "calendar_month_summary.csv", index=False)
    balanced.to_csv(
        utils.ANALYSIS_DIR / "balanced_panel_monthly.csv", index=False, date_format="%Y-%m-%d"
    )
    coverage_rows = state[
        [
            "region",
            "n_obs",
            "months_observed",
            "n_rural_obs",
            "n_urban_obs",
            "coverage_complete",
            "mean_rate",
            "weighted_rate",
        ]
    ].copy()
    coverage_rows.to_csv(utils.ANALYSIS_DIR / "coverage_by_region.csv", index=False)
    outliers[
        [
            "region",
            "date",
            "area",
            "unemployment_rate",
            "estimated_employed",
            "labour_participation_rate",
            "outlier_direction",
        ]
    ].to_csv(utils.ANALYSIS_DIR / "outliers.csv", index=False, date_format="%Y-%m-%d")
    df1.nlargest(15, "unemployment_rate")[
        ["region", "date", "area", "unemployment_rate", "estimated_employed", "labour_participation_rate"]
    ].to_csv(
        utils.ANALYSIS_DIR / "highest_observations.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    df1[df1["zero_unemployment"]][
        ["region", "date", "area", "unemployment_rate", "estimated_employed", "labour_participation_rate"]
    ].to_csv(
        utils.ANALYSIS_DIR / "zero_unemployment_observations.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    utils.save_json(utils.ANALYSIS_DIR / "eda_summary.json", summary)
    utils.save_json(
        utils.ANALYSIS_DIR / "monthly_summary.json",
        {"dataset1_monthly": monthly_records},
    )

    # Console narrative based on computed values.
    print("EXPLORATORY ANALYSIS")
    print("=" * 72)
    print(
        f"Dataset 1: {overall['observations']} observations, {overall['regions']} regions, "
        f"{overall['months']} months ({overall['date_range'][0]} to {overall['date_range'][1]})"
    )
    print(
        f"Overall weighted unemployment rate: {overall['weighted_rate']:.2f}% | "
        f"unweighted mean: {overall['unweighted_mean']:.2f}% | median: {overall['median']:.2f}%"
    )
    print(
        f"Range: {overall['min']:.2f}% to {overall['max']:.2f}%; "
        f"zero-rate observations retained: {overall['zero_unemployment_observations']}"
    )
    print(
        f"Peak month: {peak['month']} at {peak['weighted_rate']:.2f}% weighted "
        f"({peak['unweighted_rate']:.2f}% unweighted)"
    )
    print("\nHighest average states (weighted):")
    for row in top_bottom["highest"]:
        print(f"  {row['region']}: {row['weighted_rate']:.2f}%")
    print("Lowest average states (weighted):")
    for row in top_bottom["lowest"]:
        print(f"  {row['region']}: {row['weighted_rate']:.2f}%")
    print("\nArea summary:")
    for row in area_records:
        print(
            f"  {row['area']}: weighted {row['weighted_rate']:.2f}% | "
            f"unweighted {row['unweighted_rate']:.2f}% | average LPR {row['avg_lpr']:.2f}%"
        )
    print("\nPearson correlations (pooled observations):")
    print(pearson.round(3).to_string())
    print(
        f"\nIQR outliers retained: {outlier_summary['outlier_count']} "
        f"({outlier_summary['outlier_share_pct']:.1f}% of observations); "
        f"upper fence {outlier_summary['upper_fence']:.2f}%"
    )
    print(
        f"Balanced-panel check: maximum weighted-rate difference "
        f"{coverage['balanced_panel_check']['max_abs_weighted_difference_pp']:.2f} pp"
    )
    print(
        "Seasonality: insufficient evidence. Only May and June appear in both years, "
        "and 2020 repeats coincide with COVID-19."
    )
    print(f"\nSaved analysis outputs to: {utils.ANALYSIS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
