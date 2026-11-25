"""Professional charts for the Unemployment Analysis project.

Each function computes its own aggregates from the cleaned data, saves a PNG in
outputs/charts/, and returns the matplotlib figure so the notebook can display
it inline. No chart uses hard-coded result values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

utils.setup_plot_style()

MONTHLY_TICK_INTERVAL = 2


# ---------------------------------------------------------------------------
# Internal aggregation and formatting helpers
# ---------------------------------------------------------------------------
def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly weighted and unweighted aggregates for one dataset."""
    records: list[dict] = []
    for month, group in df.groupby("month_start", sort=True):
        records.append(
            {
                "month_start": month,
                "n_obs": len(group),
                "n_regions": group["region"].nunique(),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "weighted_rate": utils.weighted_rate(group),
                "median_rate": group["unemployment_rate"].median(),
                "p25": group["unemployment_rate"].quantile(0.25),
                "p75": group["unemployment_rate"].quantile(0.75),
                "total_employed": group["estimated_employed"].sum(),
                "avg_lpr": group["labour_participation_rate"].mean(),
            }
        )
    return pd.DataFrame(records)


def _state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Region-level weighted and simple averages."""
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
        total_unemployed=("_unemployed", "sum"),
        total_labour_force=("_labour_force", "sum"),
        avg_employed=("estimated_employed", "mean"),
        avg_lpr=("labour_participation_rate", "mean"),
    )
    summary["weighted_rate"] = (
        summary["total_unemployed"] / summary["total_labour_force"] * 100
    )
    return summary.reset_index()


def _period_subset(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[df["date"].between(pd.Timestamp(start), pd.Timestamp(end))]


def _format_month_axis(ax: plt.Axes, interval: int = MONTHLY_TICK_INTERVAL) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")


def _shade_covid(
    ax: plt.Axes,
    start: str = "2020-04-01",
    end: str = "2020-06-30",
    label: str | None = None,
) -> None:
    ax.axvspan(
        pd.Timestamp(start),
        pd.Timestamp(end),
        color=utils.COLORS["red"],
        alpha=0.08,
        linewidth=0,
        label=label,
        zorder=0,
    )


def _source_note(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.005,
        -0.02,
        text,
        ha="left",
        va="top",
        fontsize=8,
        color=utils.COLORS["gray"],
        wrap=True,
    )


def _finish(fig: plt.Figure, filename: str) -> Path:
    return utils.save_figure(fig, filename)


# ---------------------------------------------------------------------------
# EDA charts
# ---------------------------------------------------------------------------
def plot_overall_unemployment_trend(df1: pd.DataFrame) -> plt.Figure:
    monthly = _monthly_summary(df1)
    peak = monthly.loc[monthly["weighted_rate"].idxmax()]
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    _shade_covid(ax, label="COVID-19 shock window (Apr-Jun 2020)")
    ax.plot(
        monthly["month_start"],
        monthly["weighted_rate"],
        color=utils.COLORS["navy"],
        linewidth=2.6,
        marker="o",
        markersize=5.5,
        label="Labour-force-weighted rate",
        zorder=4,
    )
    ax.plot(
        monthly["month_start"],
        monthly["unweighted_rate"],
        color=utils.COLORS["orange"],
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Unweighted mean of region-area rates",
        zorder=3,
    )
    ax.annotate(
        f"Peak: {peak['weighted_rate']:.1f}%\n{peak['month_start']:%b %Y}",
        xy=(peak["month_start"], peak["weighted_rate"]),
        xytext=(-55, 24),
        textcoords="offset points",
        fontsize=10,
        color=utils.COLORS["red"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=utils.COLORS["red"], lw=1.2),
    )
    ax.set_title("National unemployment rate over time, May 2019 - June 2020")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_ylim(0, max(monthly["unweighted_rate"].max(), monthly["weighted_rate"].max()) + 8)
    _format_month_axis(ax)
    ax.legend(loc="upper left")
    _source_note(
        fig,
        "Weighted rate = summed estimated unemployed / summed estimated labour force "
        "across reporting observations. Unweighted mean gives every region-area unit "
        "equal weight. Shaded area is an analytical window, not an official definition.",
    )
    _finish(fig, "01_overall_unemployment_trend.png")
    return fig


def plot_state_unemployment_bar(df1: pd.DataFrame) -> plt.Figure:
    state = _state_summary(df1).sort_values("weighted_rate", ascending=True)
    national = utils.weighted_rate(df1)
    top_three = set(state.nlargest(3, "weighted_rate")["region"])
    colors = [
        utils.COLORS["red"] if region in top_three else utils.COLORS["blue"]
        for region in state["region"]
    ]
    fig, ax = plt.subplots(figsize=(11.5, 10))
    bars = ax.barh(
        state["region"],
        state["weighted_rate"],
        color=colors,
        edgecolor="white",
        height=0.72,
    )
    for bar, value in zip(bars, state["weighted_rate"]):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8.5,
            color="#263238",
        )
    ax.axvline(
        national,
        color=utils.COLORS["orange"],
        linestyle="--",
        linewidth=1.6,
        label=f"National weighted average: {national:.2f}%",
    )
    ax.set_title("Average unemployment rate by state, May 2019 - June 2020")
    ax.set_xlabel("Labour-force-weighted unemployment rate (%)")
    ax.set_ylabel("State / union territory")
    ax.set_xlim(0, state["weighted_rate"].max() + 5)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="lower right")
    _source_note(
        fig,
        "Weighted using each state's derived labour force across all available months. "
        "Top three states are highlighted. Incomplete state histories use observed months.",
    )
    _finish(fig, "02_state_unemployment_bar.png")
    return fig


def plot_rural_urban_comparison(df1: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), width_ratios=[0.85, 1.65])
    ax1, ax2 = axes

    area_stats = []
    for area in ["Rural", "Urban"]:
        subset = df1[df1["area"] == area]
        area_stats.append(
            {
                "area": area,
                "weighted": utils.weighted_rate(subset),
                "unweighted": subset["unemployment_rate"].mean(),
            }
        )
    stats = pd.DataFrame(area_stats)
    positions = np.arange(len(stats))
    width = 0.34
    bars_w = ax1.bar(
        positions - width / 2,
        stats["weighted"],
        width,
        color=[utils.AREA_COLORS[a] for a in stats["area"]],
        label="Weighted",
    )
    bars_u = ax1.bar(
        positions + width / 2,
        stats["unweighted"],
        width,
        color=[utils.AREA_COLORS[a] for a in stats["area"]],
        alpha=0.42,
        hatch="//",
        edgecolor="white",
        label="Unweighted",
    )
    for bars in (bars_w, bars_u):
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
                color="#263238",
            )
    ax1.set_xticks(positions, stats["area"])
    ax1.set_ylim(0, max(stats["weighted"].max(), stats["unweighted"].max()) + 4)
    ax1.set_ylabel("Unemployment rate (%)")
    ax1.set_title("Average level by area")
    ax1.legend(loc="upper left", fontsize=9)

    for area in ["Rural", "Urban"]:
        subset = df1[df1["area"] == area]
        monthly = _monthly_summary(subset)
        ax2.plot(
            monthly["month_start"],
            monthly["weighted_rate"],
            color=utils.AREA_COLORS[area],
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            label=area,
        )
    _shade_covid(ax2)
    ax2.set_title("Monthly weighted rate, Rural versus Urban")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Unemployment rate (%)")
    ax2.set_ylim(0, 32)
    _format_month_axis(ax2)
    ax2.legend(loc="upper left")
    fig.suptitle(
        "Rural and urban unemployment: urban areas start higher, both spike in 2020",
        fontsize=15,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        "Area rates are labour-force-weighted within each area. The shaded band marks "
        "Apr-Jun 2020. Urban rates are generally higher, but the COVID increase is large in both.",
    )
    _finish(fig, "03_rural_urban_comparison.png")
    return fig


def plot_monthly_dispersion(df1: pd.DataFrame) -> plt.Figure:
    monthly = _monthly_summary(df1)
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    _shade_covid(ax, label="Apr-Jun 2020")
    ax.fill_between(
        monthly["month_start"],
        monthly["p25"],
        monthly["p75"],
        color=utils.COLORS["blue"],
        alpha=0.18,
        label="Interquartile range across region-area rates",
    )
    ax.plot(
        monthly["month_start"],
        monthly["median_rate"],
        color=utils.COLORS["orange"],
        linewidth=2,
        marker="s",
        markersize=4,
        label="Median region-area rate",
    )
    ax.plot(
        monthly["month_start"],
        monthly["weighted_rate"],
        color=utils.COLORS["navy"],
        linewidth=2.6,
        marker="o",
        markersize=5,
        label="Labour-force-weighted national rate",
    )
    ax.set_title("Monthly unemployment and cross-region dispersion")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_ylim(0, monthly["p75"].max() + 14)
    _format_month_axis(ax)
    ax.legend(loc="upper left", fontsize=9)
    _source_note(
        fig,
        "The band shows the middle 50% of all region-area observations in each month. "
        "Widening dispersion means the shock was uneven across places, even when the "
        "national average moved less.",
    )
    _finish(fig, "04_monthly_unemployment_dispersion.png")
    return fig


def plot_correlation_heatmap(df1: pd.DataFrame) -> plt.Figure:
    columns = [
        "unemployment_rate",
        "estimated_employed",
        "labour_participation_rate",
    ]
    labels = [
        "Unemployment\nrate",
        "Estimated\nemployed",
        "Labour participation\nrate",
    ]
    corr = df1[columns].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.82},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Correlation between numerical indicators")
    _source_note(
        fig,
        "Pooled region-area-month Pearson correlations (n=740). Association is not "
        "causation; state size and repeated observations influence the coefficients. "
        "Spearman values are reported in outputs/analysis/.",
    )
    _finish(fig, "05_correlation_heatmap.png")
    return fig


def plot_unemployment_distribution(df1: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    sns.histplot(
        data=df1,
        x="unemployment_rate",
        hue="area",
        bins=32,
        multiple="layer",
        stat="count",
        alpha=0.42,
        palette=utils.AREA_COLORS,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    mean_value = df1["unemployment_rate"].mean()
    median_value = df1["unemployment_rate"].median()
    ax.axvline(
        mean_value,
        color=utils.COLORS["navy"],
        linestyle="-",
        linewidth=1.8,
        label=f"Mean: {mean_value:.2f}%",
    )
    ax.axvline(
        median_value,
        color=utils.COLORS["orange"],
        linestyle="--",
        linewidth=1.8,
        label=f"Median: {median_value:.2f}%",
    )
    ax.set_title("Distribution of unemployment rate observations")
    ax.set_xlabel("Unemployment rate (%)")
    ax.set_ylabel("Number of region-area-month observations")
    ax.legend(loc="upper right")
    _source_note(
        fig,
        "The distribution is right-skewed: most observations are below 15%, while a "
        "small number of COVID-period and small-territory observations exceed 40%. "
        "No observations were removed as outliers.",
    )
    _finish(fig, "06_unemployment_distribution.png")
    return fig


def plot_boxplot(df1: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), width_ratios=[0.8, 1.2])
    ax1, ax2 = axes
    sns.boxplot(
        data=df1,
        x="area",
        y="unemployment_rate",
        hue="area",
        palette=utils.AREA_COLORS,
        width=0.45,
        showfliers=True,
        fliersize=4,
        linewidth=1.1,
        legend=False,
        ax=ax1,
    )
    sns.stripplot(
        data=df1,
        x="area",
        y="unemployment_rate",
        color="#455A64",
        size=2,
        alpha=0.22,
        jitter=0.25,
        ax=ax1,
    )
    ax1.set_title("By area")
    ax1.set_xlabel("")
    ax1.set_ylabel("Unemployment rate (%)")

    order = ["Pre-COVID", "Transition", "COVID period"]
    palette = utils.PERIOD_COLORS
    sns.boxplot(
        data=df1,
        x="period",
        y="unemployment_rate",
        hue="period",
        order=order,
        hue_order=order,
        palette=palette,
        width=0.5,
        showfliers=True,
        fliersize=4,
        linewidth=1.1,
        legend=False,
        ax=ax2,
    )
    sns.stripplot(
        data=df1,
        x="period",
        y="unemployment_rate",
        order=order,
        color="#455A64",
        size=2,
        alpha=0.18,
        jitter=0.25,
        ax=ax2,
    )
    counts = df1["period"].value_counts()
    ax2.set_xticks(
        range(len(order)),
        [f"{label}\n(n={counts.get(label, 0)})" for label in order],
    )
    ax2.set_title("By COVID-19 period")
    ax2.set_xlabel("")
    ax2.set_ylabel("Unemployment rate (%)")
    fig.suptitle(
        "Box plots reveal high-end outliers and the April-May 2020 shock",
        fontsize=15,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        "Boxes show the median and interquartile range; whiskers extend to 1.5 x IQR "
        "and dots show individual observations (including outliers). The Mar 2020 "
        "observation is labelled Transition because the national lockdown began on 25 March.",
    )
    _finish(fig, "07_unemployment_boxplot.png")
    return fig


def plot_state_month_heatmap(df1: pd.DataFrame) -> plt.Figure:
    state = _state_summary(df1).nlargest(15, "weighted_rate")
    top_states = state["region"].tolist()
    pivot = (
        df1[df1["region"].isin(top_states)]
        .pivot_table(
            index="region",
            columns="month",
            values="unemployment_rate",
            aggfunc="mean",
        )
        .reindex(top_states)
    )
    xlabels = [pd.Timestamp(f"{month}-01").strftime("%b %y") for month in pivot.columns]
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Mean unemployment rate (%)", "shrink": 0.85},
        xticklabels=xlabels,
        yticklabels=True,
        ax=ax,
        annot_kws={"fontsize": 7.3},
    )
    if "2020-04" in list(pivot.columns):
        covid_index = list(pivot.columns).index("2020-04")
        ax.axvline(covid_index - 0.5, color=utils.COLORS["navy"], linestyle="--", linewidth=1.6)
    ax.set_title(
        "Monthly unemployment heatmap: 15 states with the highest average rates",
        fontsize=13.5,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("State / union territory")
    ax.tick_params(axis="x", labelsize=8.5)
    _source_note(
        fig,
        "Cells are means of Rural and Urban observations for that state-month; blank "
        "cells are unreported state-months and were not imputed. The dashed line marks "
        "April 2020. States peaked at different times, and not every high-average state "
        "experienced the same shock.",
    )
    _finish(fig, "08_state_month_heatmap.png")
    return fig


def plot_employment_trend(df1: pd.DataFrame) -> plt.Figure:
    months = df1["month"].nunique()
    areas = df1["area"].nunique()
    counts = df1.groupby("region").size()
    complete_states = counts[counts == months * areas].index.tolist()
    panel = df1[df1["region"].isin(complete_states)].copy()

    monthly_total = (
        panel.groupby("month_start")["estimated_employed"].sum().div(1_000_000)
    )
    monthly_area = (
        panel.groupby(["month_start", "area"])["estimated_employed"]
        .sum()
        .div(1_000_000)
        .unstack("area")
    )
    baseline_months = monthly_total[
        monthly_total.index.isin([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")])
    ]
    baseline = baseline_months.mean()
    april = monthly_total.loc[pd.Timestamp("2020-04-01")]
    april_change = (april / baseline - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), width_ratios=[1.45, 1])
    ax1, ax2 = axes
    _shade_covid(ax1, label="Apr-Jun 2020")
    ax1.plot(
        monthly_total.index,
        monthly_total.values,
        color=utils.COLORS["navy"],
        linewidth=2.6,
        marker="o",
        markersize=5,
    )
    ax1.annotate(
        f"April 2020: {april:.1f} million\n({april_change:.1f}% vs Jan-Feb average)",
        xy=(pd.Timestamp("2020-04-01"), april),
        xytext=(-125, 32),
        textcoords="offset points",
        fontsize=9.5,
        color=utils.COLORS["red"],
        arrowprops=dict(arrowstyle="->", color=utils.COLORS["red"], lw=1.1),
    )
    ax1.set_title("Total estimated employment, balanced panel")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Estimated employed (millions)")
    ax1.set_ylim(0, monthly_total.max() * 1.15)
    _format_month_axis(ax1)
    ax1.legend(loc="lower left")

    base = monthly_area.loc[pd.Timestamp("2020-02-01")]
    indexed = monthly_area.div(base, axis=1) * 100
    for area in ["Rural", "Urban"]:
        ax2.plot(
            indexed.index,
            indexed[area],
            color=utils.AREA_COLORS[area],
            linewidth=2.4,
            marker="o",
            markersize=4,
            label=area,
        )
    ax2.axhline(100, color=utils.COLORS["gray"], linestyle=":", linewidth=1)
    _shade_covid(ax2)
    ax2.set_title("Employment index by area (Feb 2020 = 100)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Index")
    _format_month_axis(ax2, interval=3)
    for label in ax2.get_xticklabels():
        label.set_fontsize(9)
    ax2.legend(loc="lower left")

    fig.suptitle(
        "COVID-19 employment loss and partial recovery",
        fontsize=15,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        f"Balanced panel: {len(complete_states)} states with all {months} months and both "
        "areas (Rural + Urban). Excludes eight states with incomplete coverage so the "
        "trend is not driven by changing sample composition. Values are survey estimates.",
    )
    _finish(fig, "09_employment_trend.png")
    return fig


def plot_labour_participation(df1: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for area in ["Rural", "Urban"]:
        monthly = _monthly_summary(df1[df1["area"] == area])
        ax.plot(
            monthly["month_start"],
            monthly["avg_lpr"],
            color=utils.AREA_COLORS[area],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=f"{area} - area mean",
        )
    national_lpr = []
    for month, group in df1.groupby("month_start", sort=True):
        national_lpr.append(
            {"month_start": month, "lpr": utils.weighted_lpr(group)}
        )
    lpr_df = pd.DataFrame(national_lpr)
    ax.plot(
        lpr_df["month_start"],
        lpr_df["lpr"],
        color=utils.COLORS["navy"],
        linewidth=2.5,
        linestyle="--",
        label="National weighted approximation",
    )
    _shade_covid(ax, label="Apr-Jun 2020")
    april = lpr_df[lpr_df["month_start"] == pd.Timestamp("2020-04-01")]["lpr"].iloc[0]
    pre = lpr_df[lpr_df["month_start"] < pd.Timestamp("2020-03-01")]["lpr"].mean()
    ax.annotate(
        f"April 2020: {april:.1f}%\n({april - pre:+.1f} pp vs pre-COVID mean)",
        xy=(pd.Timestamp("2020-04-01"), april),
        xytext=(25, -38),
        textcoords="offset points",
        fontsize=9.5,
        color=utils.COLORS["red"],
        arrowprops=dict(arrowstyle="->", color=utils.COLORS["red"], lw=1.1),
    )
    ax.set_title("Labour participation rate over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Labour participation rate (%)")
    ax.set_ylim(0, 55)
    _format_month_axis(ax)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    _source_note(
        fig,
        "LPR = labour force / population. Area lines are simple means; the national "
        "line is approximated from derived labour force and implied population because "
        "population counts are not supplied. April's drop reflects both job loss and "
        "withdrawal from the labour force.",
    )
    _finish(fig, "10_labour_participation_trend.png")
    return fig


# ---------------------------------------------------------------------------
# COVID and supplementary charts
# ---------------------------------------------------------------------------
def plot_covid_period_comparison(df1: pd.DataFrame) -> plt.Figure:
    periods = [
        ("Pre-COVID\nJan-Feb 2020", "2020-01-01", "2020-02-29"),
        ("Transition\nMar 2020", "2020-03-01", "2020-03-31"),
        ("COVID period\nApr-Jun 2020", "2020-04-01", "2020-06-30"),
        ("Peak\nApr-May 2020", "2020-04-01", "2020-05-31"),
    ]
    national = []
    for label, start, end in periods:
        subset = _period_subset(df1, start, end)
        national.append(
            {
                "period": label,
                "weighted": utils.weighted_rate(subset),
                "n": len(subset),
            }
        )
    national_df = pd.DataFrame(national)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), width_ratios=[1.3, 1])
    ax1, ax2 = axes
    colors = [
        utils.COLORS["blue"],
        utils.COLORS["gold"],
        utils.COLORS["red"],
        utils.COLORS["orange"],
    ]
    bars = ax1.bar(
        national_df["period"],
        national_df["weighted"],
        color=colors,
        width=0.62,
        edgecolor="white",
    )
    for bar, value, n in zip(bars, national_df["weighted"], national_df["n"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.55,
            f"{value:.1f}%\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#263238",
        )
    ax1.set_ylim(0, national_df["weighted"].max() + 6)
    ax1.set_ylabel("Labour-force-weighted unemployment rate (%)")
    ax1.set_title("National rate by analytical period")

    baseline = _period_subset(df1, "2020-01-01", "2020-02-29")
    covid = _period_subset(df1, "2020-04-01", "2020-06-30")
    groups = [("All", baseline, covid)]
    for area in ["Rural", "Urban"]:
        groups.append(
            (
                area,
                baseline[baseline["area"] == area],
                covid[covid["area"] == area],
            )
        )
    positions = np.arange(len(groups))
    width = 0.34
    base_vals = [utils.weighted_rate(g[1]) for g in groups]
    covid_vals = [utils.weighted_rate(g[2]) for g in groups]
    ax2.bar(
        positions - width / 2,
        base_vals,
        width,
        color=utils.COLORS["blue"],
        label="Pre-COVID (Jan-Feb 2020)",
    )
    ax2.bar(
        positions + width / 2,
        covid_vals,
        width,
        color=utils.COLORS["red"],
        label="COVID (Apr-Jun 2020)",
    )
    for pos, pre_value, covid_value in zip(positions, base_vals, covid_vals):
        ax2.text(
            pos + width / 2,
            covid_value + 0.55,
            f"+{covid_value - pre_value:.1f} pp",
            ha="center",
            fontsize=9,
            color=utils.COLORS["red"],
            fontweight="bold",
        )
    ax2.set_xticks(positions, [g[0] for g in groups])
    ax2.set_ylim(0, max(covid_vals) + 5)
    ax2.set_ylabel("Unemployment rate (%)")
    ax2.set_title("Before versus during: all, rural and urban")
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "COVID-19 period comparison: a large, broad increase in unemployment",
        fontsize=15,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        "Jan-Feb 2020 is the immediate pre-COVID baseline; March is shown separately "
        "because the national lockdown began on 25 March. Changes are percentage-point "
        "(pp) differences, not causal effects. The periods are analytical choices.",
    )
    _finish(fig, "11_covid_period_comparison.png")
    return fig


def _state_period_changes(df1: pd.DataFrame) -> pd.DataFrame:
    baseline = _period_subset(df1, "2020-01-01", "2020-02-29")
    covid = _period_subset(df1, "2020-04-01", "2020-06-30")
    rows = []
    for region in sorted(df1["region"].unique()):
        base = baseline[baseline["region"] == region]
        crisis = covid[covid["region"] == region]
        if base.empty or crisis.empty:
            continue
        base_rate = utils.weighted_rate(base)
        covid_rate = utils.weighted_rate(crisis)
        rows.append(
            {
                "region": region,
                "baseline_rate": base_rate,
                "covid_rate": covid_rate,
                "change_pp": covid_rate - base_rate,
                "change_pct": (covid_rate / base_rate - 1) * 100 if base_rate > 0 else np.nan,
                "baseline_obs": len(base),
                "covid_obs": len(crisis),
            }
        )
    return pd.DataFrame(rows).sort_values("change_pp", ascending=False)


def plot_covid_state_increases(df1: pd.DataFrame) -> plt.Figure:
    changes = _state_period_changes(df1).nlargest(12, "change_pp").sort_values("change_pp")
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(len(changes))
    ax.hlines(
        positions,
        changes["baseline_rate"],
        changes["covid_rate"],
        color="#B0BEC5",
        linewidth=2,
        zorder=2,
    )
    ax.scatter(
        changes["baseline_rate"],
        positions,
        color=utils.COLORS["blue"],
        s=55,
        label="Pre-COVID (Jan-Feb 2020)",
        zorder=3,
    )
    ax.scatter(
        changes["covid_rate"],
        positions,
        color=utils.COLORS["red"],
        s=55,
        label="COVID period (Apr-Jun 2020)",
        zorder=3,
    )
    for pos, row in zip(positions, changes.itertuples()):
        ax.text(
            max(row.baseline_rate, row.covid_rate) + 1.2,
            pos,
            f"+{row.change_pp:.1f} pp",
            va="center",
            fontsize=9,
            color=utils.COLORS["red"],
            fontweight="bold",
        )
    ax.set_yticks(positions, changes["region"])
    ax.set_xlabel("Labour-force-weighted unemployment rate (%)")
    ax.set_ylabel("State / union territory")
    ax.set_xlim(0, changes["covid_rate"].max() + 10)
    ax.set_title("Largest state-level unemployment increases, Jan-Feb to Apr-Jun 2020")
    ax.legend(loc="lower right")
    _source_note(
        fig,
        "The 12 largest absolute percentage-point increases. States with incomplete "
        "coverage use available observations. Small territories such as Puducherry can "
        "be volatile because their baseline and COVID windows contain only a few observations; "
        "absolute pp changes are more reliable here than relative percentage changes.",
    )
    _finish(fig, "12_covid_state_increases.png")
    return fig


def _d2_monthly(df2: pd.DataFrame) -> pd.DataFrame:
    records = []
    for month, group in df2.groupby("month_start", sort=True):
        records.append(
            {
                "month_start": month,
                "n_obs": len(group),
                "weighted_rate": utils.weighted_rate(group),
                "unweighted_rate": group["unemployment_rate"].mean(),
                "total_employed_m": group["estimated_employed"].sum() / 1_000_000,
                "avg_lpr": group["labour_participation_rate"].mean(),
            }
        )
    return pd.DataFrame(records)


def plot_extended_recovery_d2(df2: pd.DataFrame) -> plt.Figure:
    monthly = _d2_monthly(df2)
    january = monthly.loc[monthly["month_start"] == pd.Timestamp("2020-01-01")].iloc[0]
    peak = monthly.loc[monthly["weighted_rate"].idxmax()]
    october = monthly.loc[monthly["month_start"] == pd.Timestamp("2020-10-01")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), width_ratios=[1.35, 1])
    ax1, ax2 = axes
    _shade_covid(
        ax1,
        start="2020-04-01",
        end="2020-05-31",
        label="Apr-May 2020",
    )
    ax1.plot(
        monthly["month_start"],
        monthly["weighted_rate"],
        color=utils.COLORS["navy"],
        linewidth=2.6,
        marker="o",
        markersize=5,
        label="Weighted rate",
    )
    ax1.plot(
        monthly["month_start"],
        monthly["unweighted_rate"],
        color=utils.COLORS["orange"],
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Unweighted mean",
    )
    ax1.axhline(
        january["weighted_rate"],
        color=utils.COLORS["gray"],
        linestyle=":",
        linewidth=1.1,
        label=f"Jan 2020 level: {january['weighted_rate']:.1f}%",
    )
    ax1.annotate(
        f"Peak: {peak['weighted_rate']:.1f}%",
        xy=(peak["month_start"], peak["weighted_rate"]),
        xytext=(-20, 22),
        textcoords="offset points",
        fontsize=9.5,
        color=utils.COLORS["red"],
        arrowprops=dict(arrowstyle="->", color=utils.COLORS["red"], lw=1.1),
    )
    ax1.annotate(
        f"Oct 2020: {october['weighted_rate']:.1f}%",
        xy=(october["month_start"], october["weighted_rate"]),
        xytext=(-50, 24),
        textcoords="offset points",
        fontsize=9.5,
        color=utils.COLORS["navy"],
        arrowprops=dict(arrowstyle="->", color=utils.COLORS["navy"], lw=1.0),
    )
    ax1.set_title("Unemployment rate extended through October 2020")
    ax1.set_xlabel("2020")
    ax1.set_ylabel("Unemployment rate (%)")
    ax1.set_ylim(0, monthly["unweighted_rate"].max() + 6)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.legend(loc="upper right", fontsize=9)

    bars = ax2.bar(
        monthly["month_start"],
        monthly["total_employed_m"],
        width=20,
        color=utils.COLORS["blue"],
        alpha=0.75,
        label="Total employed (left axis)",
    )
    ax2.set_ylabel("Estimated employed (millions)", color=utils.COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=utils.COLORS["blue"])
    ax2.set_ylim(0, monthly["total_employed_m"].max() * 1.12)
    ax2b = ax2.twinx()
    ax2b.plot(
        monthly["month_start"],
        monthly["avg_lpr"],
        color=utils.COLORS["orange"],
        linewidth=2.3,
        marker="o",
        markersize=4,
        label="Average LPR (right axis)",
    )
    ax2b.set_ylabel("Labour participation rate (%)", color=utils.COLORS["orange"])
    ax2b.tick_params(axis="y", labelcolor=utils.COLORS["orange"])
    ax2b.set_ylim(30, 50)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.set_title("Employment and participation did not fully recover")
    ax2.set_xlabel("2020")
    for label in ax2.get_xticklabels():
        label.set_rotation(0)
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=8.5)

    fig.suptitle(
        "Supplementary 2020 view: unemployment falls back, but jobs and participation lag",
        fontsize=14.5,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        "Dataset 2, January-October 2020. It has one observation per state and no Rural/Urban "
        "split, so it supplements rather than merges with Dataset 1. Coverage is 26 states in "
        "Jan-Feb and 27 thereafter (Sikkim missing early months). October employment remains "
        f"{(october['total_employed_m'] / january['total_employed_m'] - 1) * 100:.1f}% versus January, "
        f"while the reported rate is {october['weighted_rate'] - january['weighted_rate']:+.1f} pp different.",
    )
    _finish(fig, "13_extended_recovery_d2.png")
    return fig


def plot_zone_unemployment(df2: pd.DataFrame) -> plt.Figure:
    records = []
    for zone, group in df2.groupby("zone", observed=True):
        records.append(
            {
                "zone": zone,
                "weighted_rate": utils.weighted_rate(group),
                "mean_rate": group["unemployment_rate"].mean(),
                "n": len(group),
            }
        )
    zone_summary = pd.DataFrame(records).sort_values("weighted_rate")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), width_ratios=[0.8, 1.5])
    ax1, ax2 = axes
    palette = sns.color_palette("Blues", n_colors=len(zone_summary) + 2)[2:]
    bars = ax1.barh(
        zone_summary["zone"],
        zone_summary["weighted_rate"],
        color=palette,
        edgecolor="white",
    )
    for bar, value in zip(bars, zone_summary["weighted_rate"]):
        ax1.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=10,
            color="#263238",
        )
    ax1.set_xlim(0, zone_summary["weighted_rate"].max() + 3)
    ax1.set_xlabel("Weighted unemployment rate (%)")
    ax1.set_ylabel("Geographic zone")
    ax1.set_title("Average by geographic zone")

    for zone in sorted(df2["zone"].unique()):
        subset = df2[df2["zone"] == zone]
        monthly = _d2_monthly(subset)
        ax2.plot(
            monthly["month_start"],
            monthly["weighted_rate"],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=zone,
        )
    _shade_covid(ax2, start="2020-04-01", end="2020-05-31")
    ax2.set_title("Monthly weighted rate by zone")
    ax2.set_xlabel("2020")
    ax2.set_ylabel("Unemployment rate (%)")
    ax2.set_ylim(0, 36)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.legend(loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(
        "Geographic-zone view from the supplementary dataset",
        fontsize=15,
        color=utils.COLORS["navy"],
        fontweight="bold",
    )
    _source_note(
        fig,
        "Zones are supplied by Dataset 2 (North, South, East, West, Northeast). They "
        "aggregate states of very different sizes and labour markets; zone averages are "
        "descriptive, not a substitute for state-level policy targeting.",
    )
    _finish(fig, "14_zone_unemployment.png")
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
EDA_CHARTS = [
    plot_overall_unemployment_trend,
    plot_state_unemployment_bar,
    plot_rural_urban_comparison,
    plot_monthly_dispersion,
    plot_correlation_heatmap,
    plot_unemployment_distribution,
    plot_boxplot,
    plot_state_month_heatmap,
    plot_employment_trend,
    plot_labour_participation,
]

COVID_CHARTS = [
    plot_covid_period_comparison,
    plot_covid_state_increases,
    plot_extended_recovery_d2,
    plot_zone_unemployment,
]


def make_eda_charts(df1: pd.DataFrame) -> list[Path]:
    """Generate all descriptive charts and return their paths."""
    paths = []
    for chart_function in EDA_CHARTS:
        fig = chart_function(df1)
        paths.append(utils.CHART_DIR / _expected_filename(chart_function.__name__))
        plt.close(fig)
    return paths


def make_covid_charts(df1: pd.DataFrame, df2: pd.DataFrame) -> list[Path]:
    """Generate COVID and supplementary charts and return their paths."""
    paths = []
    for chart_function in COVID_CHARTS[:2]:
        fig = chart_function(df1)
        paths.append(utils.CHART_DIR / _expected_filename(chart_function.__name__))
        plt.close(fig)
    for chart_function in COVID_CHARTS[2:]:
        fig = chart_function(df2)
        paths.append(utils.CHART_DIR / _expected_filename(chart_function.__name__))
        plt.close(fig)
    return paths


def _expected_filename(function_name: str) -> str:
    mapping = {
        "plot_overall_unemployment_trend": "01_overall_unemployment_trend.png",
        "plot_state_unemployment_bar": "02_state_unemployment_bar.png",
        "plot_rural_urban_comparison": "03_rural_urban_comparison.png",
        "plot_monthly_dispersion": "04_monthly_unemployment_dispersion.png",
        "plot_correlation_heatmap": "05_correlation_heatmap.png",
        "plot_unemployment_distribution": "06_unemployment_distribution.png",
        "plot_boxplot": "07_unemployment_boxplot.png",
        "plot_state_month_heatmap": "08_state_month_heatmap.png",
        "plot_employment_trend": "09_employment_trend.png",
        "plot_labour_participation": "10_labour_participation_trend.png",
        "plot_covid_period_comparison": "11_covid_period_comparison.png",
        "plot_covid_state_increases": "12_covid_state_increases.png",
        "plot_extended_recovery_d2": "13_extended_recovery_d2.png",
        "plot_zone_unemployment": "14_zone_unemployment.png",
    }
    return mapping[function_name]
