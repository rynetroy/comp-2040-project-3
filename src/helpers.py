"""
Helper functions for Downtown Winnipeg structural change analysis.

Author: Troy Dela Rosa
Course: COMP-2040
"""

import pandas as pd


def preview_dataset(
    df: pd.DataFrame,
    title: str,
    preview_cols: list[str] | None = None,
    n: int = 3
) -> None:
    """
    Print a quick overview of a dataset.

    Parameters:
        df: DataFrame to inspect.
        title: Label shown at the top of the preview.
        preview_cols: Optional list of columns to show in sample rows.
        n: Number of rows to preview.
    """
    print(f"=== {title} ===")
    print(f"Shape: {df.shape}")
    print()

    print("Columns and types:")
    print(df.dtypes.to_string())
    print()

    print("Missing values per column:")
    print(df.isnull().sum().to_string())
    print()

    print(f"First {n} rows:")
    if preview_cols is not None:
        print(df[preview_cols].head(n).to_string(index=False))
    else:
        print(df.head(n).to_string(index=False))


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with lowercase, underscore-style column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def filter_downtown(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Downtown business license rows."""
    df = df.copy()
    return df[df["community_characterization_area"] == "Downtown"].copy()


def create_is_closed(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary closed flag from business license status."""
    df = df.copy()
    closed_statuses = {"Closed (L)", "Ceased Operation", "Cancelled", "Vacant"}
    df["is_closed"] = df["status"].isin(closed_statuses)
    return df


def summarize_business_license_overview(df: pd.DataFrame) -> None:
    """
    Print a raw business-license overview after column cleaning and downtown filtering.
    """
    print("=== BUSINESS LICENSES (downtown only) ===")
    print(f"Downtown rows:   {len(df)}")
    print()

    print("Columns:")
    print(list(df.columns))
    print()

    print("Status values:")
    print(df["status"].value_counts().to_string())
    print()

    print("Issue Date sample (raw format):")
    print(df["issue_date"].head(3).to_string())


def summarize_source_registry(df: pd.DataFrame) -> None:
    """Print a summary of the source provenance registry."""
    print(f"Total sources: {len(df)}")
    print(f"  Core inputs (directly feed model): {df['usage_type'].eq('core_input').sum()}")
    print(f"  Model-ready:                       {df['model_ready'].sum()}")
    print(f"  Corroboration sources:             {df['is_corroboration'].sum()}")

    print("\nBy source category:")
    print(df["source_category"].value_counts().to_string())


def assign_phase(year: int) -> str:
    """Assign structural phase from year."""
    if year < 2015:
        return "Pre-2015 establishment"
    if year <= 2020:
        return "2015–2020 transition"
    return "2020+ restructuring"


def clean_gantt(df: pd.DataFrame, today: str = "2026-04-13") -> pd.DataFrame:
    """Clean Gantt dataset and add analysis fields."""
    df = df.copy()

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    today_ts = pd.Timestamp(today)
    df["end_date"] = df["end_date"].fillna(today_ts)

    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    df["year_start"] = df["start_date"].dt.year
    df["year_end"] = df["end_date"].dt.year
    df["phase"] = df["year_start"].apply(assign_phase)

    impact_map = {
        "Growth": "Growth",
        "Infrastructure": "Growth",
        "Adaptive Reuse": "Growth",
        "Transition": "Transition",
        "Policy": "Transition",
    }
    df["impact"] = df["category"].map(impact_map).fillna("Transition")

    completed_kw = ["Completed", "Adopted", "Demolished", "Active"]
    df["is_done"] = df["status"].apply(
        lambda s: any(kw in str(s) for kw in completed_kw)
    )

    return df


def compute_units_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Add midpoint unit estimate from low/high bounds."""
    df = df.copy()
    df["units_mid"] = (df["units_low"] + df["units_high"]) / 2
    return df


def clean_housing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean housing dataset and return full and model-ready versions."""
    df = df.copy()
    df = compute_units_mid(df)
    housing_model = df[df["include_in_model"]].copy()
    return df, housing_model


def clean_business_licenses(df: pd.DataFrame) -> pd.DataFrame:
    """Clean business license dataset for downtown analysis."""
    df = clean_column_names(df)
    df = filter_downtown(df)

    df["issue_date"] = pd.to_datetime(
        df["issue_date"],
        format="%Y %b %d %I:%M:%S %p",
        errors="coerce"
    )

    df = df.dropna(subset=["issue_date"]).copy()
    df["year"] = df["issue_date"].dt.year
    df = create_is_closed(df)

    return df


def prepare_housing_prediction_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Prepare the housing pipeline dataset for the prediction section.
    """
    df = df.copy()

    conf_map = {
        "high": 2,
        "medium_high": 2,
        "medium": 1,
        "low_medium": 0,
        "low": 0,
    }
    sq_map = {
        "strong": 2,
        "moderate": 1,
        "weak": 0,
    }

    df["confidence_encoded"] = df["confidence"].map(conf_map).fillna(1)
    df["source_quality_encoded"] = df["source_quality"].map(sq_map).fillna(1)

    features = [
        "units_mid",
        "completion_year_low",
        "confidence_encoded",
        "source_quality_encoded",
    ]
    target = "phase_status"

    df = df.dropna(subset=features + [target]).copy()
    return df, features, target

import numpy as np
import matplotlib.pyplot as plt


def compute_phase_composition(
    gantt: pd.DataFrame,
    phases: list[str] | None = None
) -> tuple[list[int], list[int], list[str]]:
    """
    Compute Growth and Transition event counts by phase using discrete rows only.

    Parameters:
        gantt: Cleaned Gantt DataFrame.
        phases: Optional phase order.

    Returns:
        growth_vals, trans_vals, phases
    """
    if phases is None:
        phases = [
            "Pre-2015 establishment",
            "2015–2020 transition",
            "2020+ restructuring",
        ]

    discrete = gantt[gantt["row_type"] == "discrete"].copy()

    growth_vals = [
        len(discrete[(discrete["phase"] == p) & (discrete["impact"] == "Growth")])
        for p in phases
    ]
    trans_vals = [
        len(discrete[(discrete["phase"] == p) & (discrete["impact"] == "Transition")])
        for p in phases
    ]

    return growth_vals, trans_vals, phases


def plot_phase_composition(
    growth_vals: list[int],
    trans_vals: list[int],
    phase_labels: list[str],
    growth_color: str,
    transition_color: str,
    save_path: str | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot stacked phase composition bars for Growth vs Transition events.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(phase_labels))
    w = 0.55

    ax.bar(
        x, growth_vals, w,
        color=growth_color,
        label="Growth events",
        alpha=0.9,
        edgecolor="none"
    )
    ax.bar(
        x, trans_vals, w,
        color=transition_color,
        label="Transition events",
        alpha=0.9,
        edgecolor="none",
        bottom=growth_vals
    )

    for i, (g, t) in enumerate(zip(growth_vals, trans_vals)):
        total = g + t
        ax.text(
            i, total + 0.15, f"{total} events",
            ha="center", fontsize=10, fontweight="bold", color="#e0ddd5"
        )
        if g > 0:
            ax.text(
                i, g / 2, f"{g}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#0d0f14"
            )
        if t > 0:
            ax.text(
                i, g + t / 2, f"{t}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#0d0f14"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=11)
    ax.set_ylabel("Number of Events")
    ax.set_title(
        "Phase Composition: Growth vs. Transition Events [L1 - OBSERVED]",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.legend(loc="upper left", framealpha=0.3, edgecolor="none")
    ax.set_ylim(0, max(g + t for g, t in zip(growth_vals, trans_vals)) + 2)
    ax.grid(axis="y", alpha=0.3)

    ax.text(
        0.01, 0.02,
        "Composite/process rows excluded (4): True North Square overall, "
        "Portage Place decline, Millennium Library ongoing, Heritage conversions cluster.",
        transform=ax.transAxes, fontsize=7, color="#0c0c0b", va="bottom"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax

def prepare_business_activity_index_data(
    filepath: str,
    base_year: int = 2013
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Load synthetic business activity sheets and compute 2013=100 index values.

    Parameters:
        filepath: Path to the Excel workbook.
        base_year: Base year for indexing.

    Returns:
        retail, office, food, combined summary, base_year
    """
    xl = pd.ExcelFile(filepath)

    retail = xl.parse('Retail License Flows')
    office = xl.parse('Office License Flows')
    food = xl.parse('Food & Services')
    syn = xl.parse('Combined Summary')

    syn = syn.dropna(subset=['Year']).copy()
    syn['Year'] = syn['Year'].astype(int)

    retail_base = retail.loc[retail['Year'] == base_year, 'Active Estimate'].values[0]
    office_base = office.loc[office['Year'] == base_year, 'Active Estimate'].values[0]
    food_base = food.loc[food['Year'] == base_year, 'Active Estimate'].values[0]

    retail = retail.copy()
    office = office.copy()
    food = food.copy()

    retail['idx'] = (retail['Active Estimate'] / retail_base) * 100
    office['idx'] = (office['Active Estimate'] / office_base) * 100
    food['idx'] = (food['Active Estimate'] / food_base) * 100

    return retail, office, food, syn, base_year


def plot_business_activity_index(
    retail: pd.DataFrame,
    office: pd.DataFrame,
    food: pd.DataFrame,
    syn: pd.DataFrame,
    decline_color: str,
    transition_color: str,
    growth_color: str,
    accent_color: str,
    save_path: str | None = None
) -> tuple[plt.Figure, list]:
    """
    Plot the two-panel business activity and vacancy comparison figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Top panel
    ax = axes[0]

    ax.plot(
        retail['Year'], retail['idx'],
        color=decline_color, linewidth=2.0,
        label='Retail (index)', marker='o', markersize=4
    )
    ax.plot(
        office['Year'], office['idx'],
        color=transition_color, linewidth=2.0,
        label='Office (index)', marker='s', markersize=4
    )
    ax.plot(
        food['Year'], food['idx'],
        color=growth_color, linewidth=2.0,
        label='Food & Services (index)', marker='^', markersize=4
    )

    ax.axhline(y=100, color='#8a8780', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.text(2010.2, 102, '2013 baseline (100)', fontsize=7, color='#8a8780')

    ax.axvspan(2009.5, 2014.5, alpha=0.06, color=decline_color)
    ax.axvspan(2014.5, 2021.5, alpha=0.04, color=accent_color)
    ax.axvspan(2021.5, 2024.5, alpha=0.04, color=growth_color)

    ax.text(2012, 20, 'Low confidence\n(synthetic)', ha='center', fontsize=7,
            color=decline_color, fontstyle='italic', alpha=0.8)
    ax.text(2018, 20, 'Medium confidence', ha='center', fontsize=7,
            color=accent_color, fontstyle='italic', alpha=0.8)
    ax.text(2023, 20, 'High confidence\n(open data)', ha='center', fontsize=7,
            color=growth_color, fontstyle='italic', alpha=0.8)

    for yr, lbl in [(2013, 'Portage Place\ndecline'), (2020, 'COVID\nshock'), (2023, 'Bay\ncloses')]:
        ax.axvline(x=yr, color='#847D6E', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.text(yr + 0.1, 130, lbl, fontsize=7, color='#847D6E', fontstyle='italic')

    ax.set_ylabel('Activity Index (2013 = 100)')
    ax.set_title(
        'Downtown Business Activity Index by Sector (2013 = 100) [L1/L2]',
        fontsize=12, fontweight='bold', pad=12
    )
    ax.legend(framealpha=0.3, edgecolor='none', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(syn['Year'])
    ax.set_xticklabels(syn['Year'], rotation=45, ha='right', fontsize=8)

    # Bottom panel
    ax2 = axes[1]

    ax2.fill_between(
        retail['Year'], retail['Vacancy Proxy Index'],
        alpha=0.2, color=decline_color
    )
    ax2.plot(
        retail['Year'], retail['Vacancy Proxy Index'],
        color=decline_color, linewidth=1.8, linestyle='--', alpha=0.7,
        label='Retail vacancy proxy (reconstructed)'
    )

    ax2.plot(
        office['Year'], office['Vacancy Proxy Index'],
        color=transition_color, linewidth=1.8, linestyle='--', alpha=0.6,
        label='Office vacancy proxy (reconstructed)'
    )

    cbre_years = [2016, 2019, 2022, 2023, 2024, 2025]
    cbre_rates = [0.087, 0.112, 0.157, 0.183, 0.184, 0.186]

    ax2.scatter(
        cbre_years, cbre_rates,
        color=accent_color, s=80, zorder=5,
        marker='D', label='CBRE office vacancy (confirmed)'
    )
    ax2.plot(
        cbre_years, cbre_rates,
        color=accent_color, linewidth=1.5,
        linestyle='-', alpha=0.8
    )

    ax2.set_ylabel('Rate / Proxy Index')
    ax2.set_title(
        'Vacancy Proxy vs. Confirmed CBRE Office Vacancy [L2 vs L1]',
        fontsize=12, fontweight='bold', pad=12
    )
    ax2.legend(framealpha=0.3, edgecolor='none', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(syn['Year'])
    ax2.set_xticklabels(syn['Year'], rotation=45, ha='right', fontsize=8)

    ax2.text(
        0.01, 0.02,
        'Synthetic model calibrated against CBRE vacancy series + BIZ reports. '
        'NOT official City data. Open data anchor: City of Winnipeg licenses (2021-2026).',
        transform=ax2.transAxes, fontsize=7, color='#8a8780', va='bottom'
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#0d0f14')

    return fig, axes

def prepare_vacancy_benchmark_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return Winnipeg office vacancy data and national benchmark data.

    Winnipeg includes confirmed and interpolated values.
    National benchmark includes confirmed values only.
    """
    vac = pd.DataFrame({
        "Year": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "Rate": [8.7, 9.2, 10.0, 11.2, 13.5, 14.8, 15.7, 18.3, 18.4, 18.6],
        "Confirmed": [True, False, False, True, False, False, True, True, True, True],
    })

    national = pd.DataFrame({
        "Year": [2023, 2024, 2025],
        "National_Rate": [19.4, 18.7, 18.0],
    })

    return vac, national


def plot_vacancy_benchmark(
    vac: pd.DataFrame,
    national: pd.DataFrame,
    decline_color: str,
    growth_color: str,
    accent_color: str,
    save_path: str | None = None
):
    """
    Plot Downtown Winnipeg office vacancy against the national benchmark.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    # Winnipeg line segments
    for i in range(len(vac) - 1):
        x0, y0, c0 = vac.iloc[i]["Year"], vac.iloc[i]["Rate"], vac.iloc[i]["Confirmed"]
        x1, y1, c1 = vac.iloc[i + 1]["Year"], vac.iloc[i + 1]["Rate"], vac.iloc[i + 1]["Confirmed"]
        both_confirmed = c0 and c1

        ax.plot(
            [x0, x1], [y0, y1],
            color=decline_color,
            linewidth=2.2,
            linestyle="-" if both_confirmed else "--",
            alpha=0.9 if both_confirmed else 0.45
        )

    # Winnipeg points
    for _, r in vac.iterrows():
        if r["Confirmed"]:
            ax.scatter(r["Year"], r["Rate"], color=decline_color, s=60, zorder=5)
            ax.text(
                r["Year"], r["Rate"] + 0.7, f"{r['Rate']}%",
                ha="center", fontsize=8, fontweight="bold", color="#e0ddd5"
            )
        else:
            ax.scatter(
                r["Year"], r["Rate"],
                color=decline_color, s=50, zorder=5,
                facecolors="none", edgecolors=decline_color,
                linewidths=1.2, alpha=0.5
            )
            ax.text(
                r["Year"], r["Rate"] + 0.7, f"{r['Rate']}%",
                ha="center", fontsize=8, color="#8a8780"
            )

    # National benchmark
    ax.plot(
        national["Year"], national["National_Rate"],
        color=accent_color, linewidth=2.0, linestyle="--", alpha=0.8, zorder=4
    )
    ax.scatter(
        national["Year"], national["National_Rate"],
        color=accent_color, marker="D", s=70, zorder=5
    )

    for _, r in national.iterrows():
        ax.text(
            r["Year"] - 0.05, r["National_Rate"] + 0.7,
            f"Natl: {r['National_Rate']}%",
            ha="center", fontsize=8, color=accent_color
        )

    # Gap annotations
    for _, r in national.iterrows():
        wpg = vac[vac["Year"] == r["Year"]].iloc[0]["Rate"]
        diff = r["National_Rate"] - wpg
        mid = (wpg + r["National_Rate"]) / 2

        ax.annotate(
            "", xy=(r["Year"], r["National_Rate"]), xytext=(r["Year"], wpg),
            arrowprops=dict(arrowstyle="<->", color="#8a8780", lw=1)
        )
        ax.text(
            r["Year"] + 0.25, mid,
            f"{'+' if diff > 0 else ''}{diff:.1f}pp",
            fontsize=7.5, color="#8a8780", va="center"
        )

    # Phase shading
    ax.axvspan(2015.5, 2019.5, alpha=0.03, color=decline_color)
    ax.axvspan(2019.5, 2025.5, alpha=0.03, color=growth_color)
    ax.text(
        2017.5, 23, "2015–2020 transition",
        ha="center", fontsize=7, color=decline_color, fontstyle="italic", alpha=0.6
    )
    ax.text(
        2022.5, 23, "2020+ restructuring",
        ha="center", fontsize=7, color=growth_color, fontstyle="italic", alpha=0.6
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Vacancy Rate (%)")
    ax.set_title(
        "Downtown Winnipeg Office Vacancy vs. National Benchmark [L1]",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 26)
    ax.set_xticks(vac["Year"])
    ax.grid(axis="y", alpha=0.3)

    conf_line = plt.Line2D([0], [0], color=decline_color, linewidth=2.2, label="Winnipeg (confirmed)")
    est_line = plt.Line2D([0], [0], color=decline_color, linewidth=2.2, linestyle="--", alpha=0.45,
                          label="Winnipeg (interpolated)")
    nat_line = plt.Line2D([0], [0], marker="D", color=accent_color, markerfacecolor=accent_color,
                          markersize=8, linestyle="--", linewidth=2.0,
                          label="National avg. (2023–2025, CBRE confirmed)")

    ax.legend(
        handles=[conf_line, est_line, nat_line],
        framealpha=0.3, edgecolor="none", fontsize=9, loc="upper left"
    )

    ax.text(
        0.01, 0.02,
        "Source: CBRE Q4 2023/2024/2025. No confirmed national estimates before 2023. "
        "4 Winnipeg years are interpolated estimates.",
        transform=ax.transAxes, fontsize=7.5, color="#8a8780", va="bottom"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax