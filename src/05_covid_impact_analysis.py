"""Phase 4: COVID-19 impact, recovery and pattern analysis.

Run from the project root (after cleaning):
    python src/05_covid_impact_analysis.py

Outputs
-------
outputs/analysis/covid_summary.json
outputs/analysis/covid_period_summary.csv
outputs/analysis/covid_monthly_changes.csv
outputs/analysis/covid_area_changes.csv
outputs/analysis/covid_state_changes.csv
outputs/analysis/covid_employment_balanced_panel.csv
outputs/analysis/d2_monthly_recovery.csv
outputs/analysis/d2_zone_summary.csv
outputs/analysis/dataset_overlap_validation.csv
outputs/charts/11_covid_period_comparison.png
outputs/charts/12_covid_state_increases.png
outputs/charts/13_extended_recovery_d2.png
outputs/charts/14_zone_unemployment.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts
import utils

PERIODS = {
    "pre_covid_full": ("2019-05-01", "2020-02-29"),
    "pre_covid_baseline": ("2020-01-01", "2020-02-29"),
    "transition_march": ("2020-03-01", "2020-03-31"),
    "covid_core": ("2020-04-01", "2020-06-30"),
    "lockdown_peak": ("2020-04-01", "2020-05-31"),
    "june_2020": ("2020-06-01", "2020-06-30"),
}


def subset_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[df["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def period_summary(df: pd.DataFrame, label: str, start: str, end: str) -> dict:
    subset = subset_period(df, start, end)
    if subset.empty:
        raise ValueError(f"Period {label} has no observations.")
    return {
        "label": label,
        "start": start,
        "end": end,
        "n_obs": int(len(subset)),
        "n_regions": int(subset["region"].nunique()),
        "unweighted_rate": float(subset["unemployment_rate"].mean()),
        "weighted_rate": utils.weighted_rate(subset),
        "median_rate": float(subset["unemployment_rate"].median()),
        "min_rate": float(subset["unemployment_rate"].min()),
        "max_rate": float(subset["unemployment_rate"].max()),
        "average_lpr": float(subset["labour_participation_rate"].mean()),
        "weighted_lpr": utils.weighted_lpr(subset),
        "total_employed_reported": float(subset["estimated_employed"].sum()),
        "total_labour_force": float(subset["estimated_labour_force"].sum()),
        "total_unemployed": float(subset["estimated_unemployed"].sum()),
        "zero_unemployment_obs": int(subset["zero_unemployment"].sum()),
    }


def build_period_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "pre_covid_full": "Pre-COVID full period (May 2019-Feb 2020)",
        "pre_covid_baseline": "Immediate pre-COVID baseline (Jan-Feb 2020)",
        "transition_march": "Transition month (March 2020)",
        "covid_core": "COVID period (April-June 2020)",
        "lockdown_peak": "Peak shock window (April-May 2020)",
        "june_2020": "June 2020",
    }
    rows = [
        period_summary(df, labels[key], *periods)
        for key, periods in PERIODS.items()
    ]
    return pd.DataFrame(rows)


def covid_monthly_changes(df: pd.DataFrame) -> pd.DataFrame:
    baseline = subset_period(df, *PERIODS["pre_covid_baseline"])
    baseline_rate = utils.weighted_rate(baseline)
    rows = []
    for month, group in df.groupby("month_start", sort=True):
        weighted = utils.weighted_rate(group)
        rows.append(
            {
                "month": group["month"].iloc[0],
                "month_start": month,
                "n_obs": len(group),
                "weighted_rate": weighted,
                "unweighted_rate": group["unemployment_rate"].mean(),
                "change_vs_baseline_pp": weighted - baseline_rate,
                "change_vs_baseline_pct": (weighted / baseline_rate - 1) * 100,
                "average_lpr": group["labour_participation_rate"].mean(),
            }
        )
    return pd.DataFrame(rows)


def area_changes(df: pd.DataFrame) -> pd.DataFrame:
    baseline = subset_period(df, *PERIODS["pre_covid_baseline"])
    covid = subset_period(df, *PERIODS["covid_core"])
    rows = []
    groups = [("All", baseline, covid)]
    for area in ["Rural", "Urban"]:
        groups.append(
            (
                area,
                baseline[baseline["area"] == area],
                covid[covid["area"] == area],
            )
        )
    for label, pre, crisis in groups:
        pre_weighted = utils.weighted_rate(pre)
        covid_weighted = utils.weighted_rate(crisis)
        rows.append(
            {
                "group": label,
                "baseline_weighted_rate": pre_weighted,
                "covid_weighted_rate": covid_weighted,
                "change_pp": covid_weighted - pre_weighted,
                "change_pct": (covid_weighted / pre_weighted - 1) * 100,
                "baseline_unweighted_rate": pre["unemployment_rate"].mean(),
                "covid_unweighted_rate": crisis["unemployment_rate"].mean(),
                "baseline_lpr": pre["labour_participation_rate"].mean(),
                "covid_lpr": crisis["labour_participation_rate"].mean(),
                "lpr_change_pp": crisis["labour_participation_rate"].mean()
                - pre["labour_participation_rate"].mean(),
                "baseline_obs": len(pre),
                "covid_obs": len(crisis),
            }
        )
    return pd.DataFrame(rows)


def state_changes(df: pd.DataFrame) -> pd.DataFrame:
    baseline = subset_period(df, *PERIODS["pre_covid_baseline"])
    covid = subset_period(df, *PERIODS["covid_core"])
    rows = []
    for region in sorted(df["region"].unique()):
        pre = baseline[baseline["region"] == region]
        crisis = covid[covid["region"] == region]
        if pre.empty or crisis.empty:
            continue
        pre_rate = utils.weighted_rate(pre)
        covid_rate = utils.weighted_rate(crisis)
        rows.append(
            {
                "region": region,
                "baseline_rate": pre_rate,
                "covid_rate": covid_rate,
                "change_pp": covid_rate - pre_rate,
                "change_pct": (covid_rate / pre_rate - 1) * 100 if pre_rate > 0 else np.nan,
                "baseline_obs": len(pre),
                "covid_obs": len(crisis),
                "baseline_months": pre["month"].nunique(),
                "covid_months": crisis["month"].nunique(),
                "complete_baseline_window": len(pre) == 4,
                "complete_covid_window": len(crisis) >= 4,
                "low_baseline_below_5pct": pre_rate < 5,
            }
        )
    changes = pd.DataFrame(rows).sort_values("change_pp", ascending=False)
    changes["absolute_change_rank"] = np.arange(1, len(changes) + 1)
    return changes


def balanced_employment(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    months = df["month"].nunique()
    areas = df["area"].nunique()
    counts = df.groupby("region").size()
    complete_states = counts[counts == months * areas].index.tolist()
    panel = df[df["region"].isin(complete_states)]
    monthly_area = (
        panel.groupby(["month_start", "area"])["estimated_employed"]
        .sum()
        .unstack("area")
    )
    monthly_total = monthly_area.sum(axis=1)
    table = pd.DataFrame(
        {
            "month": [
                panel[panel["month_start"] == month]["month"].iloc[0]
                for month in monthly_total.index
            ],
            "month_start": monthly_total.index,
            "total_employed": monthly_total.values,
            "rural_employed": monthly_area["Rural"].values,
            "urban_employed": monthly_area["Urban"].values,
            "n_states": len(complete_states),
        }
    )
    table["total_index_feb2020"] = (
        table["total_employed"] / table.loc[table["month"] == "2020-02", "total_employed"].iloc[0] * 100
    )
    baseline = table[table["month"].isin(["2020-01", "2020-02"])]
    baseline_total = float(baseline["total_employed"].mean())
    baseline_rural = float(baseline["rural_employed"].mean())
    baseline_urban = float(baseline["urban_employed"].mean())
    changes = {}
    for month in ["2020-03", "2020-04", "2020-05", "2020-06"]:
        row = table[table["month"] == month].iloc[0]
        changes[month] = {
            "total_employed": float(row["total_employed"]),
            "total_change_millions": float((row["total_employed"] - baseline_total) / 1e6),
            "total_change_pct": float((row["total_employed"] / baseline_total - 1) * 100),
            "rural_change_pct": float((row["rural_employed"] / baseline_rural - 1) * 100),
            "urban_change_pct": float((row["urban_employed"] / baseline_urban - 1) * 100),
        }
    summary = {
        "balanced_panel_states": len(complete_states),
        "state_names": complete_states,
        "baseline_definition": "Average of January and February 2020",
        "baseline_total_employed": baseline_total,
        "baseline_rural_employed": baseline_rural,
        "baseline_urban_employed": baseline_urban,
        "monthly_changes": changes,
        "note": (
            "Balanced panel of 20 states with all 14 months and both Rural and Urban rows. "
            "Using a fixed set of states prevents changing sample composition from "
            "amplifying or hiding the employment change."
        ),
    }
    return table, summary


def d2_monthly_summary(df2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in df2.groupby("month_start", sort=True):
        rows.append(
            {
                "month": group["month"].iloc[0],
                "month_start": month,
                "n_states": group["region"].nunique(),
                "n_obs": len(group),
                "weighted_rate": utils.weighted_rate(group),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "total_employed": group["estimated_employed"].sum(),
                "average_lpr": group["labour_participation_rate"].mean(),
            }
        )
    return pd.DataFrame(rows)


def d2_recovery(df2: pd.DataFrame) -> dict:
    monthly = d2_monthly_summary(df2)
    jan = monthly[monthly["month"] == "2020-01"].iloc[0]
    apr = monthly[monthly["month"] == "2020-04"].iloc[0]
    october = monthly[monthly["month"] == "2020-10"].iloc[0]
    peak = monthly.loc[monthly["weighted_rate"].idxmax()]
    common_states = sorted(
        set(df2.loc[df2["month"] == "2020-01", "region"])
        & set(df2.loc[df2["month"] == "2020-10", "region"])
    )
    common = df2[
        (df2["region"].isin(common_states)) & (df2["month"].isin(["2020-01", "2020-10"]))
    ]
    common_monthly = d2_monthly_summary(common)
    common_jan = common_monthly[common_monthly["month"] == "2020-01"].iloc[0]
    common_oct = common_monthly[common_monthly["month"] == "2020-10"].iloc[0]
    monthly_out = monthly.copy()
    numeric_columns = monthly_out.select_dtypes(include="number").columns
    monthly_out[numeric_columns] = monthly_out[numeric_columns].round(4)
    return {
        "monthly": monthly_out.to_dict(orient="records"),
        "january": {
            "weighted_rate": float(jan["weighted_rate"]),
            "unweighted_rate": float(jan["unweighted_rate"]),
            "total_employed": float(jan["total_employed"]),
            "average_lpr": float(jan["average_lpr"]),
            "n_states": int(jan["n_states"]),
        },
        "april": {
            "weighted_rate": float(apr["weighted_rate"]),
            "unweighted_rate": float(apr["unweighted_rate"]),
            "total_employed": float(apr["total_employed"]),
            "average_lpr": float(apr["average_lpr"]),
            "n_states": int(apr["n_states"]),
        },
        "peak_month": {
            "month": peak["month"],
            "weighted_rate": float(peak["weighted_rate"]),
            "unweighted_rate": float(peak["unweighted_rate"]),
        },
        "october": {
            "weighted_rate": float(october["weighted_rate"]),
            "unweighted_rate": float(october["unweighted_rate"]),
            "total_employed": float(october["total_employed"]),
            "average_lpr": float(october["average_lpr"]),
            "n_states": int(october["n_states"]),
        },
        "january_to_april": {
            "rate_change_pp": float(apr["weighted_rate"] - jan["weighted_rate"]),
            "employment_change_pct": float((apr["total_employed"] / jan["total_employed"] - 1) * 100),
            "lpr_change_pp": float(apr["average_lpr"] - jan["average_lpr"]),
        },
        "january_to_october": {
            "rate_change_pp": float(october["weighted_rate"] - jan["weighted_rate"]),
            "employment_change_pct": float(
                (october["total_employed"] / jan["total_employed"] - 1) * 100
            ),
            "lpr_change_pp": float(october["average_lpr"] - jan["average_lpr"]),
        },
        "common_state_january_vs_october": {
            "n_states": len(common_states),
            "states": common_states,
            "january_weighted_rate": float(common_jan["weighted_rate"]),
            "october_weighted_rate": float(common_oct["weighted_rate"]),
            "weighted_rate_change_pp": float(
                common_oct["weighted_rate"] - common_jan["weighted_rate"]
            ),
            "january_employed": float(common_jan["total_employed"]),
            "october_employed": float(common_oct["total_employed"]),
            "employment_change_pct": float(
                (common_oct["total_employed"] / common_jan["total_employed"] - 1) * 100
            ),
            "january_lpr": float(common_jan["average_lpr"]),
            "october_lpr": float(common_oct["average_lpr"]),
            "lpr_change_pp": float(common_oct["average_lpr"] - common_jan["average_lpr"]),
        },
    }


def zone_summary(df2: pd.DataFrame) -> pd.DataFrame:
    baseline = subset_period(df2, *PERIODS["pre_covid_baseline"])
    covid = subset_period(df2, *PERIODS["covid_core"])
    rows = []
    for zone in sorted(df2["zone"].unique()):
        all_zone = df2[df2["zone"] == zone]
        pre = baseline[baseline["zone"] == zone]
        crisis = covid[covid["zone"] == zone]
        rows.append(
            {
                "zone": zone,
                "n_obs": len(all_zone),
                "n_states": all_zone["region"].nunique(),
                "weighted_rate": utils.weighted_rate(all_zone),
                "unweighted_rate": all_zone["unemployment_rate"].mean(),
                "baseline_weighted_rate": utils.weighted_rate(pre),
                "covid_weighted_rate": utils.weighted_rate(crisis),
                "change_pp": utils.weighted_rate(crisis) - utils.weighted_rate(pre),
                "baseline_obs": len(pre),
                "covid_obs": len(crisis),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_rate", ascending=False)


def overlap_validation(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare, not merge, D1's derived overall state-month rate with D2."""
    d1_records = []
    for (region, month), group in df1.groupby(["region", "month"]):
        d1_records.append(
            {
                "region": region,
                "month": month,
                "d1_weighted_rate": utils.weighted_rate(group),
                "d1_n_area_rows": len(group),
            }
        )
    d1_state_month = pd.DataFrame(d1_records)
    d2_small = df2[["region", "month", "unemployment_rate"]].rename(
        columns={"unemployment_rate": "d2_rate"}
    )
    comparison = d1_state_month.merge(d2_small, on=["region", "month"], how="inner")
    comparison["difference_pp"] = comparison["d1_weighted_rate"] - comparison["d2_rate"]
    summary = {
        "comparison_rows": int(len(comparison)),
        "regions_compared": int(comparison["region"].nunique()),
        "pearson_correlation": float(
            comparison["d1_weighted_rate"].corr(comparison["d2_rate"])
        ),
        "spearman_correlation": float(
            comparison["d1_weighted_rate"].corr(comparison["d2_rate"], method="spearman")
        ),
        "mean_difference_pp": float(comparison["difference_pp"].mean()),
        "mean_absolute_difference_pp": float(comparison["difference_pp"].abs().mean()),
        "max_absolute_difference_pp": float(comparison["difference_pp"].abs().max()),
        "interpretation": (
            "Dataset 1's rural+urban labour-force-weighted rate closely matches Dataset 2's "
            "single regional rate over Jan-Jun 2020. This validates the derived weighting, "
            "but the datasets are still not concatenated because their populations and "
            "periods differ."
        ),
    }
    return comparison, summary


def seasonality_check(df1: pd.DataFrame) -> dict:
    years = sorted(df1["year"].unique().tolist())
    repeated_months = (
        df1.groupby("month_number")["year"].nunique().loc[lambda series: series >= 2].index
    )
    records = []
    for month_number in repeated_months:
        for year in years:
            subset = df1[(df1["month_number"] == month_number) & (df1["year"] == year)]
            if not subset.empty:
                records.append(
                    {
                        "month_number": int(month_number),
                        "month_name": subset["month_name"].iloc[0],
                        "year": int(year),
                        "weighted_rate": utils.weighted_rate(subset),
                        "n_obs": len(subset),
                    }
                )
    return {
        "assessment": "insufficient_evidence",
        "months_in_primary_dataset": int(df1["month"].nunique()),
        "years_covered": years,
        "overlapping_calendar_months": [int(m) for m in repeated_months],
        "overlapping_month_values": records,
        "note": (
            "A seasonality claim requires repeated observations of the same calendar "
            "months across several years. Only May and June are observed in both 2019 "
            "and 2020, and the 2020 values are dominated by the COVID-19 shock. No "
            "seasonal effect is claimed; the monthly pattern before 2020 is described "
            "only as month-to-month variation."
        ),
    }


def main() -> None:
    utils.ensure_directories()
    df1, df2 = utils.load_cleaned()

    periods = build_period_table(df1)
    monthly_changes = covid_monthly_changes(df1)
    area = area_changes(df1)
    states = state_changes(df1)
    employment_table, employment = balanced_employment(df1)
    recovery = d2_recovery(df2)
    zones = zone_summary(df2)
    overlap_table, overlap = overlap_validation(df1, df2)
    season = seasonality_check(df1)

    baseline = periods[periods["label"].str.startswith("Immediate")].iloc[0]
    covid = periods[periods["label"].str.startswith("COVID period")].iloc[0]
    peak = periods[periods["label"].str.startswith("Peak shock")].iloc[0]
    transition = periods[periods["label"].str.startswith("Transition")].iloc[0]
    pre_full = periods[periods["label"].str.startswith("Pre-COVID full")].iloc[0]
    june = periods[periods["label"].str.startswith("June 2020")].iloc[0]

    top_states = states.head(8).to_dict(orient="records")
    decreases = states[states["change_pp"] < 0].to_dict(orient="records")
    increases = states[states["change_pp"] > 0]
    top_relative = states.sort_values("change_pct", ascending=False).head(5).to_dict(
        orient="records"
    )

    headline = {
        "baseline": {
            "label": baseline["label"],
            "weighted_rate": float(baseline["weighted_rate"]),
            "unweighted_rate": float(baseline["unweighted_rate"]),
            "n_obs": int(baseline["n_obs"]),
        },
        "covid_core": {
            "label": covid["label"],
            "weighted_rate": float(covid["weighted_rate"]),
            "unweighted_rate": float(covid["unweighted_rate"]),
            "n_obs": int(covid["n_obs"]),
        },
        "peak": {
            "label": peak["label"],
            "weighted_rate": float(peak["weighted_rate"]),
            "unweighted_rate": float(peak["unweighted_rate"]),
            "n_obs": int(peak["n_obs"]),
        },
        "absolute_change_pp": float(covid["weighted_rate"] - baseline["weighted_rate"]),
        "relative_change_pct": float(
            (covid["weighted_rate"] / baseline["weighted_rate"] - 1) * 100
        ),
        "peak_change_pp": float(peak["weighted_rate"] - baseline["weighted_rate"]),
        "peak_relative_change_pct": float(
            (peak["weighted_rate"] / baseline["weighted_rate"] - 1) * 100
        ),
        "unweighted_change_pp": float(
            covid["unweighted_rate"] - baseline["unweighted_rate"]
        ),
        "transition_rate": float(transition["weighted_rate"]),
        "june_rate": float(june["weighted_rate"]),
        "pre_covid_full_rate": float(pre_full["weighted_rate"]),
    }

    states_overview = {
        "states_increased": int(len(increases)),
        "states_decreased": int(len(decreases)),
        "top_absolute_increases": top_states,
        "top_relative_increases": top_relative,
        "decreases": decreases,
    }

    causal_statement = (
        "The analysis documents a sharp, coincident change in unemployment during the "
        "COVID-19 period. It cannot establish that COVID-19 alone caused the change. "
        "There is no counterfactual, no randomised variation, and no control group; "
        "lockdowns, survey disruption, migration, fiscal support and global demand "
        "changed at the same time. The before-and-after comparison is descriptive."
    )

    result = {
        "analysis_periods": periods.round(4).to_dict(orient="records"),
        "headline": headline,
        "area_changes": area.round(4).to_dict(orient="records"),
        "state_changes_overview": states_overview,
        "employment_balanced_panel": employment,
        "dataset2_recovery": recovery,
        "zone_summary": zones.round(4).to_dict(orient="records"),
        "overlap_validation": overlap,
        "seasonality": season,
        "causality_statement": causal_statement,
        "limitations": [
            "Monthly survey estimates, not a monthly unemployment register; revisions and measurement error are possible.",
            "The primary dataset ends in June 2020, so it captures the shock and early rebound, not the full recovery.",
            "Dataset 2 extends to October 2020 but has no Rural/Urban split and one row per state.",
            "The two files are different views and are not concatenated.",
            "Small states and union territories have few observations and volatile rates; Puducherry's extreme change is a clear example.",
            "No age, gender, industry, education or district detail is available, so policy targeting cannot be more granular than state/area.",
            "Correlation and before-after comparison do not identify causal effects.",
        ],
    }

    # Save tables.
    periods.to_csv(utils.ANALYSIS_DIR / "covid_period_summary.csv", index=False)
    monthly_changes.to_csv(
        utils.ANALYSIS_DIR / "covid_monthly_changes.csv", index=False, date_format="%Y-%m-%d"
    )
    area.to_csv(utils.ANALYSIS_DIR / "covid_area_changes.csv", index=False)
    states.to_csv(utils.ANALYSIS_DIR / "covid_state_changes.csv", index=False)
    employment_table.to_csv(
        utils.ANALYSIS_DIR / "covid_employment_balanced_panel.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    recovery_df = pd.DataFrame(recovery["monthly"])
    recovery_df.to_csv(
        utils.ANALYSIS_DIR / "d2_monthly_recovery.csv", index=False, date_format="%Y-%m-%d"
    )
    zones.to_csv(utils.ANALYSIS_DIR / "d2_zone_summary.csv", index=False)
    overlap_table.to_csv(
        utils.ANALYSIS_DIR / "dataset_overlap_validation.csv", index=False
    )
    pd.DataFrame(season["overlapping_month_values"]).to_csv(
        utils.ANALYSIS_DIR / "seasonality_check.csv", index=False
    )
    utils.save_json(utils.ANALYSIS_DIR / "covid_summary.json", result)

    # Charts 11-14.
    print("Generating COVID-19 and supplementary charts...")
    generated = []
    for function in charts.COVID_CHARTS[:2]:
        fig = function(df1)
        generated.append(
            utils.CHART_DIR / charts._expected_filename(function.__name__)
        )
        utils.plt.close(fig)
    for function in charts.COVID_CHARTS[2:]:
        fig = function(df2)
        generated.append(
            utils.CHART_DIR / charts._expected_filename(function.__name__)
        )
        utils.plt.close(fig)

    print("\nCOVID-19 IMPACT")
    print("=" * 72)
    print(
        f"Baseline Jan-Feb 2020: {baseline['weighted_rate']:.2f}% weighted "
        f"({baseline['n_obs']} observations)"
    )
    print(
        f"COVID period Apr-Jun 2020: {covid['weighted_rate']:.2f}% weighted "
        f"({covid['n_obs']} observations)"
    )
    print(
        f"Change: {headline['absolute_change_pp']:+.2f} percentage points "
        f"({headline['relative_change_pct']:+.1f}%)"
    )
    print(
        f"Peak window Apr-May 2020: {peak['weighted_rate']:.2f}% "
        f"({headline['peak_change_pp']:+.2f} pp vs baseline)"
    )
    print("\nArea changes:")
    for row in area.itertuples():
        print(
            f"  {row.group}: {row.baseline_weighted_rate:.2f}% -> "
            f"{row.covid_weighted_rate:.2f}% ({row.change_pp:+.2f} pp, "
            f"{row.change_pct:+.1f}%)"
        )
    print("\nLargest absolute state increases:")
    for row in states.head(8).itertuples():
        print(
            f"  {row.region}: {row.baseline_rate:.2f}% -> {row.covid_rate:.2f}% "
            f"({row.change_pp:+.2f} pp)"
        )
    print(
        f"\nBalanced-panel employment: {employment['balanced_panel_states']} states; "
        f"April 2020 {employment['monthly_changes']['2020-04']['total_employed'] / 1e6:.1f} million, "
        f"{employment['monthly_changes']['2020-04']['total_change_pct']:+.1f}% vs Jan-Feb average"
    )
    print(
        f"Dataset 2 extension: Jan 2020 {recovery['january']['weighted_rate']:.2f}% -> "
        f"Apr 2020 {recovery['april']['weighted_rate']:.2f}% -> "
        f"Oct 2020 {recovery['october']['weighted_rate']:.2f}%"
    )
    print(
        "Oct 2020 versus Jan 2020 (same 26 states): "
        f"rate {recovery['common_state_january_vs_october']['weighted_rate_change_pp']:+.2f} pp; "
        f"employed {recovery['common_state_january_vs_october']['employment_change_pct']:+.1f}%; "
        f"LPR {recovery['common_state_january_vs_october']['lpr_change_pp']:+.2f} pp"
    )
    print(
        f"\nOverlap check (not a merge): {overlap['comparison_rows']} state-months, "
        f"r={overlap['pearson_correlation']:.3f}, "
        f"mean difference={overlap['mean_difference_pp']:+.2f} pp"
    )
    print(
        "Seasonality: insufficient evidence; only May and June repeat across years, "
        "and 2020 repeats coincide with COVID-19."
    )
    print("\nCharts:")
    for path in generated:
        if not path.exists():
            raise FileNotFoundError(f"Chart not created: {path}")
        print(f"  {path.name}")

    print(f"\nSaved: {utils.ANALYSIS_DIR / 'covid_summary.json'}")
    print(f"Saved tables in: {utils.ANALYSIS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
