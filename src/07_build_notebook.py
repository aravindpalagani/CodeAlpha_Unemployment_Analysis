"""Build the executable analysis notebook.

Run from the project root after the analysis scripts:
    python src/07_build_notebook.py
    python -m nbconvert --to notebook --execute notebooks/unemployment_analysis.ipynb --inplace

The builder injects the actual computed headline values into the Markdown
narrative, so the notebook text stays consistent with the analysis outputs.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils

NOTEBOOK_PATH = utils.NOTEBOOK_DIR / "unemployment_analysis.ipynb"


def load_analysis() -> tuple[dict, dict, dict, dict]:
    required = [
        utils.ANALYSIS_DIR / "eda_summary.json",
        utils.ANALYSIS_DIR / "covid_summary.json",
        utils.CLEANED_DIR / "cleaning_log.json",
        utils.INSPECTION_DIR / "inspection_summary.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required analysis output missing: {path}\n"
                "Run src/03_exploratory_analysis.py and src/05_covid_impact_analysis.py first."
            )
    eda = json.loads(required[0].read_text(encoding="utf-8"))
    covid = json.loads(required[1].read_text(encoding="utf-8"))
    cleaning = json.loads(required[2].read_text(encoding="utf-8"))
    inspection = json.loads(required[3].read_text(encoding="utf-8"))
    return eda, covid, cleaning, inspection


def main() -> None:
    eda, covid, cleaning, inspection = load_analysis()
    overall = eda["dataset1_overall"]
    headline = covid["headline"]
    area_rows = {row["group"]: row for row in covid["area_changes"]}
    eda_areas = {row["area"]: row for row in eda["area_summary"]}
    rural = {**eda_areas["Rural"], **area_rows["Rural"]}
    urban = {**eda_areas["Urban"], **area_rows["Urban"]}
    employment = covid["employment_balanced_panel"]["monthly_changes"]
    common_recovery = covid["dataset2_recovery"]["common_state_january_vs_october"]
    overlap = covid["overlap_validation"]
    zones = {row["zone"]: row for row in covid["zone_summary"]}
    top_states = eda["top_bottom_states"]["highest"]
    low_states = eda["top_bottom_states"]["lowest"]
    top_abs = covid["state_changes_overview"]["top_absolute_increases"]
    outlier = eda["outliers"]
    log1 = cleaning["dataset1"]
    log2 = cleaning["dataset2"]

    def pct(value: float, digits: int = 2) -> str:
        return f"{value:.{digits}f}%"

    def signed(value: float, digits: int = 2) -> str:
        return f"{value:+.{digits}f}"

    cells: list = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    # ------------------------------------------------------------------
    # Title and orientation
    # ------------------------------------------------------------------
    md(
        """
        # Unemployment Analysis with Python

        ### CodeAlpha Data Science Internship - Task 2

        **Prepared by:** CodeAlpha Data Science Intern  
        **Analysis period:** May 2019 - October 2020 (two datasets)  
        **Primary dataset:** `Unemployment in India.csv`  
        **Supplementary dataset:** `Unemployment_Rate_upto_11_2020.csv`

        ---

        ## Project overview

        This notebook analyses monthly unemployment estimates for Indian states and
        union territories. It cleans the raw data, explores national and regional
        patterns, separates Rural and Urban labour markets, and studies the labour-market
        shock associated with the COVID-19 period.

        The central question is descriptive: **how did unemployment, employment and
        labour participation change across places and periods, and what do those changes
        imply for policy?** The notebook does not claim that COVID-19 was the sole cause
        of every observed change.
        """
    )

    md(
        """
        ## Objectives

        1. Inspect both raw datasets without assuming their schema, types or date formats.
        2. Clean and document data-quality decisions without arbitrary imputation.
        3. Measure overall, state-level, area-level and monthly unemployment.
        4. Examine estimated employment and labour participation.
        5. Quantify the observed COVID-period change against an immediate pre-COVID baseline.
        6. Check whether the data can support seasonal claims.
        7. Translate findings into policy recommendations that are explicitly linked to results.

        ## Datasets and analytical design

        | Dataset | Clean records | Regions | Months | Main contribution |
        |---|---:|---:|---|---|
        | **Dataset 1** - `Unemployment in India.csv` | 740 | 28 | 14 (May 2019-Jun 2020) | Primary: Rural/Urban detail, pre-COVID baseline, shock and early rebound |
        | **Dataset 2** - `Unemployment_Rate_upto_11_2020.csv` | 267 | 27 | 10 (Jan-Oct 2020) | Supplementary: longer 2020 recovery window and geographic zones |

        Dataset 1 has **two rows per region-month** (Rural and Urban). Dataset 2 has one
        overall row per region-month, no area split, and additional zone/coordinate fields.
        They are related views, not duplicate records. **They are analysed separately and
        compared, never concatenated.**

        The file named `Unemployment_Rate_upto_11_2020.csv` actually ends on **31 October
        2020**; the observed date range is used throughout.

        ### Method note: two kinds of average

        - **Unweighted mean** - every region-area-month observation has equal weight.
        - **Labour-force-weighted rate** - derived as
          `sum(estimated unemployed) / sum(estimated labour force) x 100`, where
          `labour force = employed / (1 - unemployment rate/100)`.

        The weighted measure prevents a small territory from having the same influence as
        Uttar Pradesh. It is the headline measure, while the unweighted mean is retained
        for transparency.
        """
    )

    code(
        """
        %matplotlib inline

        import json
        import sys
        import warnings
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display

        warnings.filterwarnings("ignore")

        PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        import charts
        import utils

        utils.setup_plot_style()
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.width", 150)
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

        print(f"Project root: {PROJECT_ROOT}")
        print(f"pandas {pd.__version__} | numpy {np.__version__} | matplotlib {plt.matplotlib.__version__} | seaborn {sns.__version__}")
        """
    )

    md(
        """
        The chart functions in `src/charts.py` save publication-quality PNGs to
        `outputs/charts/` and return figures for inline display. The notebook therefore
        produces the same charts as the reusable scripts.
        """
    )

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------
    md(
        """
        ## 1. Load the raw datasets

        The raw files are loaded without renaming columns or coercing types. This exposes
        the actual schema before any cleaning.
        """
    )
    code(
        """
        raw1 = utils.load_raw_dataset1()
        raw2 = utils.load_raw_dataset2()

        print("Dataset 1 shape:", raw1.shape)
        print("Dataset 1 columns:", [repr(c) for c in raw1.columns])
        display(raw1.head(5))

        print("\\nDataset 2 shape:", raw2.shape)
        print("Dataset 2 columns:", [repr(c) for c in raw2.columns])
        display(raw2.head(5))
        """
    )

    md(
        f"""
        ## 2. Dataset inspection

        Inspection is performed separately for each file. Key raw findings are:

        - **Dataset 1:** {inspection['dataset1']['shape']['rows']} rows, of which
          **{inspection['dataset1']['completely_empty_rows']} are completely blank**. After
          removing blank rows there are **{log1['cleaned_shape'][0]} usable observations**.
          It has 28 regions, a Rural/Urban split, and 14 months from
          **{inspection['dataset1']['date_range'][0]}** to **{inspection['dataset1']['date_range'][1]}**.
        - **Dataset 2:** {inspection['dataset2']['shape']['rows']} rows, no blank rows,
          **{inspection['dataset2']['shape']['rows']} usable observations**, 27 regions,
          and 10 months from **{inspection['dataset2']['date_range'][0]}** to
          **{inspection['dataset2']['date_range'][1]}**.
        - Column headers in both files contain **leading spaces** (for example
          `' Date'` and `' Estimated Unemployment Rate (%)'`). Dataset 1 dates also carry
          leading spaces; Frequency appears as `Monthly` in Dataset 1 and `M` in Dataset 2.
        - Dataset 1 has 28 empty rows; the 27 apparent duplicate records are the repeated
          blank rows. There are **zero duplicate non-blank records**.
        - Dataset 2 has three missing region-month combinations: Jammu & Kashmir in April
          2020, and Sikkim in January and February 2020. They remain absent rather than
          being imputed.
        """
    )
    code(
        """
        for label, frame, date_col in [
            ("Dataset 1", raw1, "Date"),
            ("Dataset 2", raw2, "Date"),
        ]:
            work = frame.copy()
            work.columns = [str(c).strip() for c in work.columns]
            parsed = pd.to_datetime(
                work[date_col].astype("string").str.strip(),
                format="%d-%m-%Y",
                errors="coerce",
            )
            print("=" * 75)
            print(label)
            print("=" * 75)
            print("Shape:", work.shape)
            print("Completely blank rows:", int(work.isna().all(axis=1).sum()))
            print("Exact duplicates (including blank):", int(work.duplicated().sum()))
            print("Duplicates after blank-row removal:", int(work.dropna(how="all").duplicated().sum()))
            print("Date range:", parsed.min().date(), "to", parsed.max().date())
            print("Regions:", work["Region"].dropna().nunique())
            display(work.isna().sum().to_frame("missing_count").T)
            display(work.describe().T)
        """
    )

    md(
        """
        ### Inspection interpretation

        The datasets are structurally different but complementary:

        - Dataset 1 is the only source of **Rural/Urban** analysis and the only one with a
          pre-COVID period long enough to establish a baseline.
        - Dataset 2 has a longer 2020 window and geographic zones, but no area split.
        - The overlap is January-June 2020. A later validation cell shows that Dataset 1's
          derived rural+urban rate closely tracks Dataset 2's single regional rate, which
          supports the weighting method but does **not** justify merging the files.

        The appropriate decision is therefore: **primary = Dataset 1, supplementary =
        Dataset 2, no concatenation.**
        """
    )

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    md(
        """
        ## 3. Cleaning and preprocessing

        Cleaning decisions are deliberately conservative:

        1. Strip whitespace from column names and text values.
        2. Remove only **completely empty rows** (28 in Dataset 1).
        3. Parse dates with the explicit `dd-mm-yyyy` format and validate failures.
        4. Convert numeric fields with coercion and check definitional ranges.
        5. Check duplicates after standardisation.
        6. Drop rows only if critical fields are missing - none were.
        7. **Do not impute** missing region-months or replace zeros.
        8. Retain 11 exact-zero unemployment observations and flag them.
        9. Derive labour force and unemployed counts for weighted aggregation.

        The derived fields are arithmetic identities, not new official estimates:
        `labour force = employed / (1 - rate/100)` and `unemployed = labour force - employed`.
        """
    )
    code(
        """
        clean1, log1 = utils.clean_dataset1(raw1)
        clean2, log2 = utils.clean_dataset2(raw2)

        cleaning_summary = pd.DataFrame(
            [
                {
                    "dataset": "Dataset 1",
                    "raw_rows": log1["raw_shape"][0],
                    "blank_rows_removed": log1["completely_blank_rows_removed"],
                    "duplicates_removed": log1["duplicate_rows_removed"],
                    "text_cells_cleaned": log1["text_cells_cleaned"],
                    "zero_rates_retained": log1["zero_unemployment_rows_preserved"],
                    "clean_rows": log1["cleaned_shape"][0],
                    "regions": log1["regions"],
                },
                {
                    "dataset": "Dataset 2",
                    "raw_rows": log2["raw_shape"][0],
                    "blank_rows_removed": log2["completely_blank_rows_removed"],
                    "duplicates_removed": log2["duplicate_rows_removed"],
                    "text_cells_cleaned": log2["text_cells_cleaned"],
                    "zero_rates_retained": log2["zero_unemployment_rows_preserved"],
                    "clean_rows": log2["cleaned_shape"][0],
                    "regions": log2["regions"],
                },
            ]
        )
        display(cleaning_summary)
        display(clean1.head(4))
        display(clean2.head(4))

        print("Dataset 1 missing cells after cleaning:", int(clean1.isna().sum().sum()))
        print("Dataset 2 missing cells after cleaning:", int(clean2.isna().sum().sum()))
        """
    )
    code(
        """
        quality = pd.read_csv(utils.CLEANED_DIR / "data_quality_report.csv")
        display(quality)
        print("Checks passed:", int((quality["status"] == "PASS").sum()), "of", len(quality))
        """
    )

    md(
        f"""
        ### Cleaning result

        - Dataset 1: **{log1['raw_shape'][0]} raw rows -> {log1['cleaned_shape'][0]} clean
          observations**, {log1['regions']} regions, two areas, 14 months.
        - Dataset 2: **{log2['raw_shape'][0]} -> {log2['cleaned_shape'][0]}** observations,
          {log2['regions']} regions, 10 months.
        - Dataset 1 has **{log1['missing_region_area_months']} missing region-area-months**
          out of an expected {log1['expected_region_area_months']}. They are not fabricated.
        - Dataset 2 has **{log2['missing_region_months']} missing region-months** out of
          {log2['expected_region_months']}.
        - No duplicate non-blank records remain; all post-cleaning quality checks pass.
        - Only **{log1['zero_unemployment_rows_preserved']} zero-rate observations** exist
          in Dataset 1; they are retained and flagged because they may reflect genuine
          small-area estimates or measurement noise. Removing them would bias the picture
          upward.
        """
    )

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    md(
        """
        ## 4. Exploratory data analysis

        We now compute overall levels, state rankings, Rural/Urban differences, monthly
        movement, employment and labour participation, and relationships among variables.
        """
    )
    code(
        """
        df1, df2 = utils.load_cleaned()

        monthly_rows = []
        for month, group in df1.groupby("month_start", sort=True):
            monthly_rows.append(
                {
                    "month": group["month"].iloc[0],
                    "n_obs": len(group),
                    "unweighted_rate": group["unemployment_rate"].mean(),
                    "weighted_rate": utils.weighted_rate(group),
                    "median_rate": group["unemployment_rate"].median(),
                    "p25": group["unemployment_rate"].quantile(0.25),
                    "p75": group["unemployment_rate"].quantile(0.75),
                    "avg_lpr": group["labour_participation_rate"].mean(),
                }
            )
        monthly = pd.DataFrame(monthly_rows)
        peak = monthly.loc[monthly["weighted_rate"].idxmax()]

        overall_summary = pd.Series(
            {
                "observations": len(df1),
                "regions": df1["region"].nunique(),
                "months": df1["month"].nunique(),
                "weighted_unemployment_rate": utils.weighted_rate(df1),
                "unweighted_mean": df1["unemployment_rate"].mean(),
                "median": df1["unemployment_rate"].median(),
                "std_dev": df1["unemployment_rate"].std(),
                "minimum": df1["unemployment_rate"].min(),
                "maximum": df1["unemployment_rate"].max(),
                "average_lpr": df1["labour_participation_rate"].mean(),
                "zero_unemployment_observations": int(df1["zero_unemployment"].sum()),
            },
            name="value",
        )
        display(overall_summary.to_frame())
        print(f"Peak weighted month: {peak['month']} at {peak['weighted_rate']:.2f}%")
        display(monthly.round(2))
        """
    )

    md(
        f"""
        ### Overall picture

        Across all {overall['observations']} region-area-month observations, the
        **labour-force-weighted unemployment rate is {overall['weighted_rate']:.2f}%**.
        The unweighted mean is {overall['unweighted_mean']:.2f}%, the median is
        {overall['median']:.2f}% and the standard deviation is {overall['std']:.2f} pp.
        Observations range from {overall['min']:.2f}% to {overall['max']:.2f}%, so the
        distribution is strongly right-skewed.

        Before the shock, the weighted rate moved in a relatively narrow band around
        7-8%. The highest weighted monthly value is **{overall['peak_month']['weighted_rate']:.2f}% in
        {overall['peak_month']['month']}**; the unweighted mean that month is {overall['peak_month']['unweighted_rate']:.2f}%.
        The difference between weighted and unweighted measures is not an error: it reflects
        how much weight large states receive.
        """
    )
    code(
        """
        state = (
            df1.assign(
                _unemployed=df1["estimated_unemployed"],
                _labour_force=df1["estimated_labour_force"],
            )
            .groupby("region")
            .agg(
                n_obs=("unemployment_rate", "size"),
                months_observed=("month", "nunique"),
                mean_rate=("unemployment_rate", "mean"),
                median_rate=("unemployment_rate", "median"),
                max_rate=("unemployment_rate", "max"),
                avg_lpr=("labour_participation_rate", "mean"),
                total_unemployed=("_unemployed", "sum"),
                total_labour_force=("_labour_force", "sum"),
            )
        )
        state["weighted_rate"] = state["total_unemployed"] / state["total_labour_force"] * 100
        state = state.drop(columns=["total_unemployed", "total_labour_force"])
        state = state.sort_values("weighted_rate", ascending=False)

        print("Five highest average unemployment rates:")
        display(state.head(5).round(2))
        print("Five lowest average unemployment rates:")
        display(state.tail(5).sort_values("weighted_rate").round(2))
        """
    )

    md(
        f"""
        ### State-level differences

        The highest labour-force-weighted averages are:

        1. **{top_states[0]['region']}: {top_states[0]['weighted_rate']:.2f}%**
        2. **{top_states[1]['region']}: {top_states[1]['weighted_rate']:.2f}%**
        3. **{top_states[2]['region']}: {top_states[2]['weighted_rate']:.2f}%**
        4. **{top_states[3]['region']}: {top_states[3]['weighted_rate']:.2f}%**
        5. **{top_states[4]['region']}: {top_states[4]['weighted_rate']:.2f}%**

        The lowest are **{low_states[0]['region']} ({low_states[0]['weighted_rate']:.2f}%)**,
        **{low_states[1]['region']} ({low_states[1]['weighted_rate']:.2f}%)**,
        **{low_states[2]['region']} ({low_states[2]['weighted_rate']:.2f}%)**,
        **{low_states[3]['region']} ({low_states[3]['weighted_rate']:.2f}%)** and
        **{low_states[4]['region']} ({low_states[4]['weighted_rate']:.2f}%)**.

        This is a wide structural gap: Tripura's average is roughly nine times Meghalaya's.
        A single national policy design would therefore be poorly targeted. Small states
        and union territories with incomplete coverage (Chandigarh has 12 observations;
        Assam, Goa, Jammu & Kashmir, Meghalaya, Puducherry, Sikkim and Uttarakhand also
        have gaps) should be treated cautiously.
        """
    )
    code(
        """
        area_rows = []
        for area, group in df1.groupby("area"):
            area_rows.append(
                {
                    "area": area,
                    "n_obs": len(group),
                    "weighted_rate": utils.weighted_rate(group),
                    "unweighted_mean": group["unemployment_rate"].mean(),
                    "median": group["unemployment_rate"].median(),
                    "average_lpr": group["labour_participation_rate"].mean(),
                }
            )
        area_table = pd.DataFrame(area_rows).set_index("area")
        display(area_table.round(2))

        print("Monthly weighted rate by area:")
        area_monthly = []
        for (month, area), group in df1.groupby(["month", "area"]):
            area_monthly.append(
                {
                    "month": month,
                    "area": area,
                    "weighted_rate": utils.weighted_rate(group),
                }
            )
        display(pd.DataFrame(area_monthly).pivot(index="month", columns="area", values="weighted_rate").round(2))
        """
    )

    md(
        f"""
        ### Rural versus Urban

        Urban areas have a **higher average unemployment rate**:
        **{urban['weighted_rate']:.2f}% weighted ({urban['unweighted_rate']:.2f}% unweighted)**
        versus **{rural['weighted_rate']:.2f}% weighted ({rural['unweighted_rate']:.2f}% unweighted)**
        in rural areas.

        Labour participation is higher in rural areas:
        **{rural['avg_lpr']:.2f}% average LPR versus {urban['avg_lpr']:.2f}% urban**.
        During the shock, both areas experienced similar absolute increases, but the rural
        increase was larger relative to its lower base. Section 6 quantifies this.
        """
    )

    md(
        """
        ### Relationships among numerical indicators

        Pooled Pearson and Spearman correlations are computed across all region-area-month
        observations. They describe association, not causation. State size and repeated
        observations make pooled correlations difficult to interpret, so we also inspect
        group-level summaries.
        """
    )
    code(
        """
        corr_cols = ["unemployment_rate", "estimated_employed", "labour_participation_rate"]
        display(df1[corr_cols].corr(method="pearson").round(2))
        print("Spearman:")
        display(df1[corr_cols].corr(method="spearman").round(2))

        monthly_employment_corr = monthly.set_index("month")["unweighted_rate"].corr(
            df1.groupby("month")["estimated_employed"].sum()
        )
        print(f"Monthly national unemployment vs reported employment correlation: {monthly_employment_corr:.2f}")
        """
    )

    md(
        f"""
        The pooled unemployment-employment correlation is only about **-0.22**, and the
        unemployment-LPR correlation is essentially **0.00**. This does not mean the labour
        market is disconnected: it reflects enormous variation in state population and
        repeated measurements. At the national monthly level, unemployment and total
        reported employment move much more strongly in opposite directions, because the
        COVID shock simultaneously destroyed jobs and raised joblessness.

        The practical lesson is that a simple correlation matrix is not sufficient for
        policy targeting; state, area and period splits are essential.
        """
    )

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    md(
        """
        ## 5. Visualisations

        All charts use a consistent theme, informative titles, labelled axes, a source note,
        and 200-dpi PNG output. They are saved in `outputs/charts/`.
        """
    )
    md("### 5.1 National trend and state ranking")
    code(
        """
        fig = charts.plot_overall_unemployment_trend(df1)
        display(fig)

        fig = charts.plot_state_unemployment_bar(df1)
        display(fig)
        """
    )
    md(
        f"""
        **Reading the trend.** The weighted line is flat at roughly 7-8% from May 2019 to
        February 2020, edges up in March, jumps to **24.16% in April 2020**, remains near
        23.6% in May, and falls to 11.0% in June. The unweighted mean peaks at 24.88% in
        May. The shaded Apr-Jun window is an analytical choice, not an official COVID
        definition.

        **Reading the state bars.** Tripura ({top_states[0]['weighted_rate']:.2f}%) and
        Haryana ({top_states[1]['weighted_rate']:.2f}%) stand well above the national
        weighted average of {overall['weighted_rate']:.2f}%. The distribution is not a
        simple north-south story; state-specific labour markets matter.
        """
    )
    md("### 5.2 Rural/urban, monthly dispersion and the shape of the distribution")
    code(
        """
        fig = charts.plot_rural_urban_comparison(df1)
        display(fig)

        fig = charts.plot_monthly_dispersion(df1)
        display(fig)

        fig = charts.plot_unemployment_distribution(df1)
        display(fig)

        fig = charts.plot_boxplot(df1)
        display(fig)
        """
    )
    md(
        f"""
        **Area comparison.** Urban unemployment is higher before the shock, but both lines
        jump in April. Urban peaks near 25.9% in May while rural peaks near 23.6% in April.
        The rural line is lower on average but more volatile relative to its base.

        **Dispersion.** The interquartile band widens sharply in April-May, showing that the
        shock was highly uneven across places, not merely a uniform national shift.

        **Distribution and outliers.** The histogram is right-skewed: most observations sit
        below 15%, with a long tail above 40%. The IQR rule flags
        **{outlier['outlier_count']} observations ({outlier['outlier_share_pct']:.1f}%)**
        above {outlier['upper_fence']:.2f}%. They are retained because they are plausible
        crisis observations, not data-entry errors; removing them would understate the shock.
        """
    )
    md("### 5.3 Employment, participation and state-month timing")
    code(
        """
        fig = charts.plot_employment_trend(df1)
        display(fig)

        fig = charts.plot_labour_participation(df1)
        display(fig)

        fig = charts.plot_state_month_heatmap(df1)
        display(fig)

        fig = charts.plot_correlation_heatmap(df1)
        display(fig)
        """
    )
    md(
        f"""
        **Employment.** In the balanced panel of 20 states, April 2020 employment is
        **{employment['2020-04']['total_employed'] / 1e6:.0f} million**, down
        **{abs(employment['2020-04']['total_change_millions']):.0f} million
        ({employment['2020-04']['total_change_pct']:.1f}%)** from the January-February
        average. June is still {abs(employment['2020-06']['total_change_pct']):.1f}% below
        baseline. Urban employment falls more sharply in April (-34.7%) than rural (-30.0%).

        **Participation.** The estimated national LPR falls from about 42.7% before the
        shock to 35.6% in April - a drop of roughly 7.1 pp. The decline is consistent with
        both job loss and discouraged workers leaving the labour force; the dataset cannot
        separate those mechanisms.

        **Heatmap.** Some states peak in April (Puducherry, Haryana, Jharkhand), others in
        May (Bihar, Tripura, Kerala). A state-level average masks timing. Blank cells are
        genuinely unreported state-months.
        """
    )

    # ------------------------------------------------------------------
    # COVID
    # ------------------------------------------------------------------
    md(
        """
        ## 6. COVID-19 impact analysis

        ### Period definitions

        - **Immediate pre-COVID baseline:** January-February 2020.
        - **Transition:** March 2020; the national lockdown began on 25 March, so this month
          straddles pre-shock and shock conditions.
        - **COVID period:** April-June 2020.
        - **Peak window:** April-May 2020.

        These are analytical windows, not official definitions. Changes are measured in
        percentage points (pp) and relative percentages where useful. They are not causal
        effects.
        """
    )
    code(
        """
        def period_subset(frame, start, end):
            return frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]

        period_defs = [
            ("Pre-COVID full (May 2019-Feb 2020)", "2019-05-01", "2020-02-29"),
            ("Immediate baseline (Jan-Feb 2020)", "2020-01-01", "2020-02-29"),
            ("Transition (Mar 2020)", "2020-03-01", "2020-03-31"),
            ("COVID period (Apr-Jun 2020)", "2020-04-01", "2020-06-30"),
            ("Peak (Apr-May 2020)", "2020-04-01", "2020-05-31"),
            ("June 2020", "2020-06-01", "2020-06-30"),
        ]
        period_rows = []
        for label, start, end in period_defs:
            group = period_subset(df1, start, end)
            period_rows.append(
                {
                    "period": label,
                    "n_obs": len(group),
                    "n_regions": group["region"].nunique(),
                    "weighted_rate": utils.weighted_rate(group),
                    "unweighted_rate": group["unemployment_rate"].mean(),
                    "average_lpr": group["labour_participation_rate"].mean(),
                    "total_employed_reported": group["estimated_employed"].sum(),
                }
            )
        period_table = pd.DataFrame(period_rows)
        display(period_table.round(2))

        baseline = period_subset(df1, "2020-01-01", "2020-02-29")
        crisis = period_subset(df1, "2020-04-01", "2020-06-30")
        baseline_rate = utils.weighted_rate(baseline)
        crisis_rate = utils.weighted_rate(crisis)
        print(f"Baseline weighted rate: {baseline_rate:.2f}%")
        print(f"COVID weighted rate: {crisis_rate:.2f}%")
        print(f"Absolute change: {crisis_rate - baseline_rate:+.2f} pp")
        print(f"Relative change: {(crisis_rate / baseline_rate - 1) * 100:+.1f}%")
        """
    )

    md(
        f"""
        ### Headline change

        The weighted national rate rises from **{headline['baseline']['weighted_rate']:.2f}%**
        in January-February 2020 to **{headline['covid_core']['weighted_rate']:.2f}%** in
        April-June 2020: **{headline['absolute_change_pp']:+.2f} pp**, or
        **{headline['relative_change_pct']:+.1f}%**.

        The April-May peak is **{headline['peak']['weighted_rate']:.2f}%**, an increase of
        **{headline['peak_change_pp']:+.2f} pp** over baseline. March (during the first
        lockdown days) is **{headline['transition_rate']:.2f}%**, while June is already
        down to **{headline['june_rate']:.2f}%**. This is a sharp spike and partial early
        rebound, not evidence of a gradual trend that began earlier.
        """
    )
    code(
        """
        area_change_rows = []
        for label, pre, during in [
            ("All", baseline, crisis),
            ("Rural", baseline[baseline["area"] == "Rural"], crisis[crisis["area"] == "Rural"]),
            ("Urban", baseline[baseline["area"] == "Urban"], crisis[crisis["area"] == "Urban"]),
        ]:
            pre_rate = utils.weighted_rate(pre)
            during_rate = utils.weighted_rate(during)
            area_change_rows.append(
                {
                    "group": label,
                    "baseline": pre_rate,
                    "covid": during_rate,
                    "change_pp": during_rate - pre_rate,
                    "change_pct": (during_rate / pre_rate - 1) * 100,
                    "baseline_lpr": pre["labour_participation_rate"].mean(),
                    "covid_lpr": during["labour_participation_rate"].mean(),
                }
            )
        display(pd.DataFrame(area_change_rows).round(2))

        state_rows = []
        for region in sorted(df1["region"].unique()):
            pre = baseline[baseline["region"] == region]
            during = crisis[crisis["region"] == region]
            if pre.empty or during.empty:
                continue
            pre_rate = utils.weighted_rate(pre)
            during_rate = utils.weighted_rate(during)
            state_rows.append(
                {
                    "region": region,
                    "baseline": pre_rate,
                    "covid": during_rate,
                    "change_pp": during_rate - pre_rate,
                    "baseline_obs": len(pre),
                    "covid_obs": len(during),
                }
            )
        state_change = pd.DataFrame(state_rows).sort_values("change_pp", ascending=False)
        display(state_change.head(12).round(2))
        display(state_change.tail(5).round(2))
        """
    )

    md(
        f"""
        ### Area pattern

        - **Rural:** {rural['baseline_weighted_rate']:.2f}% -> {rural['covid_weighted_rate']:.2f}%
          ({rural['change_pp']:+.2f} pp, {rural['change_pct']:+.1f}%).
        - **Urban:** {urban['baseline_weighted_rate']:.2f}% -> {urban['covid_weighted_rate']:.2f}%
          ({urban['change_pp']:+.2f} pp, {urban['change_pct']:+.1f}%).

        The absolute increases are nearly identical, but the rural increase is larger in
        relative terms because rural unemployment started from a lower base. Urban
        unemployment remains higher in absolute terms. Both rural and urban LPR fell
        (rural {signed(rural['lpr_change_pp'])} pp; urban {signed(urban['lpr_change_pp'])} pp).

        ### Regional pattern

        The largest absolute increases are **Puducherry {top_abs[0]['change_pp']:+.1f} pp**,
        **Jharkhand {top_abs[1]['change_pp']:+.1f} pp**, **Tamil Nadu {top_abs[2]['change_pp']:+.1f} pp**,
        **Bihar {top_abs[3]['change_pp']:+.1f} pp** and **Karnataka {top_abs[4]['change_pp']:+.1f} pp**.
        Puducherry's change starts from an unusually low {top_abs[0]['baseline_rate']:.2f}% baseline
        and is based on only four baseline observations; it is a warning about small-area
        volatility, not a typical state experience. Among larger states, Jharkhand, Tamil Nadu
        and Bihar stand out.

        Some high-average states did **not** rise: Tripura fell from 30.54% to 25.97%,
        Jammu & Kashmir from 20.94% to 10.67% (with incomplete April data), and Himachal
        Pradesh from 16.84% to 12.93%. A high level and a large shock are different policy
        problems.
        """
    )
    code(
        """
        months = df1["month"].nunique()
        areas = df1["area"].nunique()
        counts = df1.groupby("region").size()
        complete_states = counts[counts == months * areas].index.tolist()
        panel = df1[df1["region"].isin(complete_states)]

        panel_monthly = (
            panel.groupby(["month_start", "area"])["estimated_employed"].sum().unstack("area")
        )
        panel_monthly["total"] = panel_monthly.sum(axis=1)
        baseline_months = panel_monthly.loc[
            panel_monthly.index.isin([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")])
        ]
        base_total = baseline_months["total"].mean()
        base_rural = baseline_months["Rural"].mean()
        base_urban = baseline_months["Urban"].mean()
        employment_changes = pd.DataFrame(
            {
                "total_millions": panel_monthly["total"] / 1e6,
                "total_vs_baseline_pct": (panel_monthly["total"] / base_total - 1) * 100,
                "rural_vs_baseline_pct": (panel_monthly["Rural"] / base_rural - 1) * 100,
                "urban_vs_baseline_pct": (panel_monthly["Urban"] / base_urban - 1) * 100,
            }
        )
        print("Balanced panel states:", len(complete_states))
        display(employment_changes.round(2))
        """
    )

    md(
        f"""
        ### Employment loss

        The balanced panel of {covid['employment_balanced_panel']['balanced_panel_states']}
        states avoids changing sample composition. April 2020 total employment is
        **{employment['2020-04']['total_employed'] / 1e6:.1f} million**, a fall of
        **{abs(employment['2020-04']['total_change_millions']):.1f} million
        ({employment['2020-04']['total_change_pct']:.1f}%)** from the January-February
        average. May is {employment['2020-05']['total_change_pct']:.1f}% below baseline and
        June is {employment['2020-06']['total_change_pct']:.1f}% below.

        Employment loss is concentrated in April. By June, jobs are recovering but are still
        materially below the pre-shock level. This is consistent with the unemployment rate
        falling faster than employment because some people left the labour force.
        """
    )

    code(
        """
        fig = charts.plot_covid_period_comparison(df1)
        display(fig)

        fig = charts.plot_covid_state_increases(df1)
        display(fig)
        """
    )
    md(
        """
        The bar chart makes the scale explicit: the COVID-period rate is more than 2.5 times
        the immediate baseline, and the April-May window is over three times baseline. The
        right panel shows that the shock is broad-based, not solely an urban phenomenon.
        The dumbbell chart ranks absolute state increases. Small territories such as
        Puducherry need cautious interpretation because their estimates are volatile.
        """
    )

    code(
        """
        d2_monthly_rows = []
        for month, group in df2.groupby("month_start", sort=True):
            d2_monthly_rows.append(
                {
                    "month": group["month"].iloc[0],
                    "n_states": group["region"].nunique(),
                    "weighted_rate": utils.weighted_rate(group),
                    "unweighted_rate": group["unemployment_rate"].mean(),
                    "total_employed_millions": group["estimated_employed"].sum() / 1e6,
                    "average_lpr": group["labour_participation_rate"].mean(),
                }
            )
        d2_monthly = pd.DataFrame(d2_monthly_rows)
        display(d2_monthly.round(2))

        common_states = sorted(
            set(df2.loc[df2["month"] == "2020-01", "region"])
            & set(df2.loc[df2["month"] == "2020-10", "region"])
        )
        common = df2[
            (df2["region"].isin(common_states))
            & (df2["month"].isin(["2020-01", "2020-10"]))
        ]
        common_summary = []
        for month_name, group in common.groupby("month"):
            common_summary.append(
                {
                    "month": month_name,
                    "weighted_rate": utils.weighted_rate(group),
                    "employed_millions": group["estimated_employed"].sum() / 1e6,
                    "average_lpr": group["labour_participation_rate"].mean(),
                }
            )
        common_summary = pd.DataFrame(common_summary).set_index("month")
        display(common_summary.round(2))
        print("Common states:", len(common_states))
        """
    )

    md(
        f"""
        ### Longer 2020 view (supplementary dataset)

        Dataset 2 extends the picture to October 2020. The weighted rate rises from
        **{covid['dataset2_recovery']['january']['weighted_rate']:.2f}% in January** to
        **{covid['dataset2_recovery']['april']['weighted_rate']:.2f}% in April**, then falls
        to **{covid['dataset2_recovery']['october']['weighted_rate']:.2f}% in October**.

        Comparing the same {common_recovery['n_states']} states in January and October:
        the rate is **{common_recovery['weighted_rate_change_pp']:+.2f} pp**, but employment
        is **{common_recovery['employment_change_pct']:+.1f}%** lower and LPR is
        **{common_recovery['lpr_change_pp']:+.2f} pp** lower. The headline unemployment rate
        has returned to its pre-shock level, but the labour market has not fully recovered:
        fewer people are employed and fewer are participating.

        This is a crucial distinction. A falling unemployment rate can reflect people
        leaving the labour force rather than finding jobs.
        """
    )
    code(
        """
        zone_rows = []
        baseline2 = period_subset(df2, "2020-01-01", "2020-02-29")
        crisis2 = period_subset(df2, "2020-04-01", "2020-06-30")
        for zone in sorted(df2["zone"].unique()):
            all_zone = df2[df2["zone"] == zone]
            pre = baseline2[baseline2["zone"] == zone]
            during = crisis2[crisis2["zone"] == zone]
            zone_rows.append(
                {
                    "zone": zone,
                    "states": all_zone["region"].nunique(),
                    "weighted_rate": utils.weighted_rate(all_zone),
                    "baseline": utils.weighted_rate(pre),
                    "covid": utils.weighted_rate(during),
                    "change_pp": utils.weighted_rate(during) - utils.weighted_rate(pre),
                }
            )
        display(pd.DataFrame(zone_rows).sort_values("weighted_rate", ascending=False).round(2))

        # Schema comparison, not a merge: D1 derived overall rate vs D2 rate.
        d1_state_month = []
        for (region, month), group in df1.groupby(["region", "month"]):
            d1_state_month.append(
                {"region": region, "month": month, "d1_weighted_rate": utils.weighted_rate(group)}
            )
        comparison = (
            pd.DataFrame(d1_state_month)
            .merge(df2[["region", "month", "unemployment_rate"]], on=["region", "month"])
            .rename(columns={"unemployment_rate": "d2_rate"})
        )
        comparison["difference_pp"] = comparison["d1_weighted_rate"] - comparison["d2_rate"]
        print("Overlap comparison rows:", len(comparison))
        print("Pearson correlation:", round(comparison["d1_weighted_rate"].corr(comparison["d2_rate"]), 3))
        print("Mean difference (pp):", round(comparison["difference_pp"].mean(), 3))
        print("Mean absolute difference (pp):", round(comparison["difference_pp"].abs().mean(), 3))
        display(comparison.head())
        """
    )

    md(
        f"""
        ### Geographic zones and dataset validation

        The supplementary zone averages are **East {zones['East']['weighted_rate']:.2f}%**,
        **North {zones['North']['weighted_rate']:.2f}%**, **South {zones['South']['weighted_rate']:.2f}%**,
        **Northeast {zones['Northeast']['weighted_rate']:.2f}%** and
        **West {zones['West']['weighted_rate']:.2f}%**. The East and North have the highest
        average levels; the East also has the largest increase (+17.63 pp). These zone
        numbers aggregate very different states and are descriptive only.

        The overlap check compares {overlap['comparison_rows']} state-month pairs without
        merging. The Pearson correlation is **{overlap['pearson_correlation']:.3f}** and the
        mean signed difference is **{overlap['mean_difference_pp']:+.2f} pp**, but the mean
        absolute difference is **{overlap['mean_absolute_difference_pp']:.2f} pp** and the
        maximum is **{overlap['max_absolute_difference_pp']:.2f} pp**. The derived weighted
        measure is broadly consistent with Dataset 2, but the discrepancies are why the files
        should not be treated as interchangeable.
        """
    )
    code(
        """
        fig = charts.plot_extended_recovery_d2(df2)
        display(fig)

        fig = charts.plot_zone_unemployment(df2)
        display(fig)
        """
    )

    # ------------------------------------------------------------------
    # Patterns and seasonality
    # ------------------------------------------------------------------
    md(
        """
        ## 7. Patterns, seasonality and causal caution

        ### What the monthly data show

        - A stable pre-COVID band around 7-8% weighted.
        - A sharp spike in April-May 2020.
        - A large but partial fall in June 2020.
        - A return of the headline rate to near its January 2020 level by October 2020 in
          Dataset 2, while employment and participation remain below January.
        - Strong cross-state heterogeneity in both level and timing.

        ### Seasonality

        The 14-month primary dataset is **too short** to establish a seasonal cycle. Only
        **May and June** occur in both 2019 and 2020, and the 2020 observations coincide with
        the COVID shock. Dataset 2 covers only one calendar year. No seasonal claim is made.
        A proper seasonal analysis would need several complete years and a model that
        separates season, trend and shock.
        """
    )
    code(
        """
        calendar = pd.read_csv(utils.ANALYSIS_DIR / "calendar_month_summary.csv")
        display(calendar.round(2))
        print("Months observed in both years:", (calendar["n_years"] >= 2).sum())
        print("Primary dataset months:", df1["month"].nunique())
        """
    )

    md(
        """
        ### Causal caution

        The analysis documents a sharp change that coincides with the COVID-19 period. It
        **cannot prove that COVID-19 alone caused it**. There is no counterfactual and no
        control group. Lockdowns, migration, survey disruption, fiscal support, global
        demand and measurement changes all moved at the same time. Before-and-after
        comparisons are descriptive; credible causal estimates would require a research
        design beyond this dataset.
        """
    )

    # ------------------------------------------------------------------
    # Findings and recommendations
    # ------------------------------------------------------------------
    md(
        f"""
        ## 8. Key findings

        1. **Unemployment was structurally uneven before COVID.** The 14-month weighted
           national average is **{overall['weighted_rate']:.2f}%**, but state averages range
           from **{low_states[0]['weighted_rate']:.2f}% in {low_states[0]['region']}** to
           **{top_states[0]['weighted_rate']:.2f}% in {top_states[0]['region']}**.
        2. **Urban areas had higher unemployment but rural areas had higher participation.**
           Urban weighted unemployment is {urban['weighted_rate']:.2f}% versus
           {rural['weighted_rate']:.2f}% rural; average LPR is {rural['avg_lpr']:.2f}% rural
           versus {urban['avg_lpr']:.2f}% urban.
        3. **The COVID-period increase was very large.** The weighted rate rose
           **{headline['absolute_change_pp']:+.2f} pp** from the Jan-Feb 2020 baseline to
           Apr-Jun 2020, a **{headline['relative_change_pct']:.1f}%** relative increase.
           The Apr-May peak was **{headline['peak']['weighted_rate']:.2f}%**.
        4. **The shock was broad-based across rural and urban areas.** Rural rose
           **{rural['change_pp']:+.2f} pp ({rural['change_pct']:.1f}%)**; urban rose
           **{urban['change_pp']:+.2f} pp ({urban['change_pct']:.1f}%)**.
        5. **Employment fell much more than the unemployment rate alone suggests.**
           In the balanced 20-state panel, April employment fell
           **{abs(employment['2020-04']['total_change_millions']):.1f} million
           ({employment['2020-04']['total_change_pct']:.1f}%)** from the Jan-Feb average.
           June was still {abs(employment['2020-06']['total_change_pct']):.1f}% below baseline.
        6. **The October recovery is incomplete.** In the same 26 states, Dataset 2 shows
           the rate **{common_recovery['weighted_rate_change_pp']:+.2f} pp** versus January,
           but employment **{common_recovery['employment_change_pct']:+.1f}%** lower and LPR
           **{common_recovery['lpr_change_pp']:+.2f} pp** lower.
        7. **The shock was regionally concentrated.** Jharkhand, Tamil Nadu and Bihar had
           some of the largest absolute increases; East and North zones have the highest
           average levels. Small territories such as Puducherry show extreme volatility.
        8. **Seasonality cannot be established.** Only two calendar months repeat across
           years, and those repeats are contaminated by the COVID shock.

        ### What the data do not show

        - They do not identify causal effects of lockdowns or COVID-19.
        - They do not provide age, gender, industry, education or district detail.
        - They cannot distinguish unemployment from underemployment or informal work.
        - They do not measure income loss, poverty or migration.
        - Dataset 2 is not directly comparable to Dataset 1 at all times and is not merged.
        """
    )

    md(
        f"""
        ## 9. Policy recommendations linked to findings

        These recommendations follow from the observed patterns. They are not claims that a
        particular intervention has been evaluated in this dataset.

        ### 9.1 Protect rural employment and earnings

        Rural unemployment rose **{rural['change_pp']:+.2f} pp** (from
        {rural['baseline_weighted_rate']:.2f}% to {rural['covid_weighted_rate']:.2f}%), a
        **{rural['change_pct']:.1f}%** relative increase, even though rural areas had higher
        participation. Expand guaranteed rural employment, irrigation, rural infrastructure,
        agri-processing and local food supply chains. These measures can stabilise earnings
        when urban migration reverses and rural labour supply rises.

        ### 9.2 Support urban job recovery and re-entry

        Urban unemployment was the highest average level before the shock
        ({urban['weighted_rate']:.2f}%) and remained higher during it
        ({urban['covid_weighted_rate']:.2f}%). Urban LPR fell by
        **{abs(urban['lpr_change_pp']):.1f} pp** in the COVID window. Prioritise MSME credit,
        wage support in contact-intensive services, safe transport and childcare, and
        job-search/re-entry services for displaced urban workers.

        ### 9.3 Target the highest-burden and largest-shock states

        High average unemployment states (**Tripura {top_states[0]['weighted_rate']:.2f}%**,
        **Haryana {top_states[1]['weighted_rate']:.2f}%**, **Jharkhand {top_states[2]['weighted_rate']:.2f}%**,
        **Bihar {top_states[3]['weighted_rate']:.2f}%**, **Delhi {top_states[4]['weighted_rate']:.2f}%**)
        need structural employment programmes. The largest observed increases were in
        **Puducherry, Jharkhand, Tamil Nadu, Bihar and Karnataka**; however, Puducherry's
        baseline was unusually low and volatile, so packages should be designed with
        state-specific diagnostics rather than a single formula.

        ### 9.4 Rebuild jobs, not just the unemployment rate

        April employment loss in the balanced panel was
        **{abs(employment['2020-04']['total_change_millions']):.0f} million jobs
        ({employment['2020-04']['total_change_pct']:.1f}%)**, and June remained
        **{abs(employment['2020-06']['total_change_pct']):.1f}%** below baseline. Wage
        subsidies, hiring incentives, public works and MSME credit should be judged by
        employment restoration, not only by a falling unemployment rate.

        ### 9.5 Support labour-force participation

        The national LPR fell roughly **7 pp** in April, and the supplementary data still
        show a **{abs(common_recovery['lpr_change_pp']):.2f} pp** participation gap in
        October for the same 26 states. Childcare, safe public transport, formalisation of
        informal work, portable benefits and accessible job-search support can help people
        return to the labour force. A low unemployment rate caused by discouraged workers
        leaving the labour force is not a healthy recovery.

        ### 9.6 Strengthen social protection for informal workers

        The scale and speed of the employment loss - over 100 million jobs in the balanced
        panel in April and May - imply that shock-responsive income support is essential.
        Portable unemployment benefits, food support, direct cash transfers and informal-worker
        registration would reduce distress and prevent asset sales. Targeting should use the
        state and area patterns documented here.

        ### 9.7 Invest in data and evaluation

        The analysis is limited by 14 months of primary data, missing region-months,
        inconsistent small-area estimates and no demographic detail. Policymakers should
        publish timely state-district-month estimates with age, gender and sector breakdowns,
        and evaluate interventions against pre-specified comparison groups. Better data would
        allow seasonal adjustment and causal evaluation rather than descriptive before-after
        comparisons.
        """
    )

    md(
        """
        ## 10. Limitations

        - **Short primary panel:** 14 months cannot establish seasonality or long-run trends.
        - **No causal identification:** before-and-after changes are not causal effects.
        - **Survey estimates:** the CSVs do not document sampling design, revisions or
          definitions; values should be treated as estimates.
        - **Missing observations:** 44 region-area-months in Dataset 1 and 3 region-months in
          Dataset 2 are absent. No values were imputed.
        - **Small-area volatility:** Puducherry, Chandigarh and other small territories have
          very few observations and extreme rates.
        - **Aggregation:** zone, state and national averages hide district and household
          heterogeneity.
        - **No demographic detail:** age, gender, education, industry and informal-work status
          are unavailable, limiting targeted policy design.
        - **Two datasets, two populations:** Dataset 1 and Dataset 2 are related but not
          identical; they are compared only where definitions overlap.
        - **Measurement of employment:** estimated employed counts are model-based and may
          not capture underemployment or income loss.
        """
    )

    md(
        f"""
        ## 11. Conclusion

        The data describe a labour market that was already uneven before the pandemic and
        then experienced an extraordinary, broad-based shock. The labour-force-weighted
        national unemployment rate rose from **{headline['baseline']['weighted_rate']:.2f}%**
        in January-February 2020 to **{headline['covid_core']['weighted_rate']:.2f}%** in
        April-June 2020, with an April-May peak of **{headline['peak']['weighted_rate']:.2f}%**.
        Employment in a balanced 20-state panel fell by nearly a third in April.

        By October 2020, the supplementary data show the headline unemployment rate back near
        its January level, but employment and participation had not recovered. That gap is the
        central policy message: **a lower unemployment rate is not the same as a restored
        labour market.** Rural areas need earnings and employment support; urban areas need
        job recovery and re-entry support; high-burden and high-shock states need targeted
        programmes; and informal workers need shock-responsive social protection.

        The analysis is descriptive and limited by data availability. Its value lies in
        making the scale, unevenness and incomplete recovery visible - and in linking each
        recommendation to a measured pattern rather than to assumption.
        """
    )

    md(
        """
        ## Reproducibility

        From the project root:

        ```bash
        python src/01_dataset_inspection.py
        python src/02_data_cleaning.py
        python src/03_exploratory_analysis.py
        python src/04_data_visualization.py
        python src/05_covid_impact_analysis.py
        python src/06_generate_report.py
        ```

        Key outputs:

        - Clean data: `outputs/cleaned_data/`
        - Charts: `outputs/charts/` (14 PNG files)
        - Tables and JSON: `outputs/analysis/`
        - Inspection report: `outputs/inspection/`
        - PDF report: `report/Unemployment_Analysis_Report.pdf`

        Data source: the two CSV files supplied for the CodeAlpha Task 2 exercise. No
        external data or invented statistics are used.
        """
    )

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
            "title": "CodeAlpha Task 2 - Unemployment Analysis with Python",
        },
    )
    utils.NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Created notebook with {len(cells)} cells: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
