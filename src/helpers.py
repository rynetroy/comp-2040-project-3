"""
Helper functions for Downtown Winnipeg structural change analysis.

Author: Troy Dela Rosa
Course: COMP-2040
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

def prepare_category_phase_matrix(
    gantt: pd.DataFrame,
    phase_order: list[str] | None = None
) -> pd.DataFrame:
    """
    Build a category-by-phase count matrix from the cleaned Gantt dataset.

    Parameters:
        gantt: Cleaned Gantt DataFrame.
        phase_order: Optional ordered list of phases.

    Returns:
        Crosstab DataFrame of category counts by phase.
    """
    if phase_order is None:
        phase_order = [
            "Pre-2015 establishment",
            "2015–2020 transition",
            "2020+ restructuring",
        ]

    ct = pd.crosstab(gantt["category"], gantt["phase"])
    ct = ct.reindex(columns=phase_order, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    return ct


def plot_category_phase_matrix(
    ct: pd.DataFrame,
    save_path: str | None = None
):
    """
    Plot the category-by-phase heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(
        [
            "Pre-2015\nEstablishment",
            "2015–2020\nTransition",
            "2020+\nRestructuring",
        ],
        fontsize=10
    )
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(ct.index, fontsize=10)

    for i in range(len(ct)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            if val > 0:
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="#0d0f14" if val >= 2 else "#e0ddd5"
                )

    ax.set_title(
        "Category × Phase Distribution [L3 - INFERRED]",
        fontsize=14, fontweight="bold", pad=15
    )
    plt.colorbar(im, ax=ax, label="Event count", shrink=0.7)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax

def prepare_residential_pipeline_plot_data(
    housing: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare model-ready housing data and labels for the residential pipeline chart.
    """
    housing_model = housing[housing["include_in_model"]].copy()

    if "units_mid" not in housing_model.columns:
        housing_model["units_mid"] = (
            housing_model["units_low"] + housing_model["units_high"]
        ) / 2

    short_labels = [
        "225 Carlton\n(TNS)",
        "300 Main",
        "Portage Place\nPh.1",
        "Wehwehneh\n(Bay Bldg)",
        "Belgica Block\n+ Alloway",
        "St. Charles\nHotel",
        "Maw's Garage\n/ Sanford",
    ]

    return housing_model, short_labels


def plot_residential_pipeline(
    housing_model: pd.DataFrame,
    short_labels: list[str],
    growth_color: str,
    transition_color: str,
    accent_color: str,
    save_path: str | None = None
):
    """
    Plot the residential pipeline by project and by status share.
    """
    status_colors = {
        "Completed": growth_color,
        "Under Construction": transition_color,
        "Planned": accent_color,
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    for i, (_, row) in enumerate(housing_model.iterrows()):
        c = status_colors.get(row["phase_status"], "#7a7a7a")
        ax1.bar(i, row["units_mid"], color=c, alpha=0.85, width=0.65, edgecolor="none")

        if row["units_low"] != row["units_high"]:
            ax1.plot(
                [i, i], [row["units_low"], row["units_high"]],
                color="#e0ddd5", lw=1.5, alpha=0.6
            )
            for bound in [row["units_low"], row["units_high"]]:
                ax1.plot(
                    [i - 0.12, i + 0.12], [bound, bound],
                    color="#e0ddd5", lw=1, alpha=0.5
                )

        ax1.text(
            i, row["units_mid"] + 15, f"~{int(row['units_mid'])}",
            ha="center", fontsize=9, color="#e0ddd5"
        )

    ax1.set_xticks(range(len(housing_model)))
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylabel("Residential Units")
    ax1.set_title(
        "Residential Pipeline by Project [L1/L2]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax1.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=v, label=k) for k, v in status_colors.items()]
    ax1.legend(handles=handles, fontsize=8, framealpha=0.3, edgecolor="none")

    ax1.text(
        0.01, 0.02,
        "Model-ready rows only. Aggregate infill row (200-400 units) excluded.",
        transform=ax1.transAxes, fontsize=7, color="#8a8780", va="bottom"
    )

    status_totals = housing_model.groupby("phase_status")["units_mid"].sum()
    colors_pie = [status_colors.get(s, "#7a7a7a") for s in status_totals.index]

    wedges, texts, autotexts = ax2.pie(
        status_totals.values,
        labels=status_totals.index,
        colors=colors_pie,
        autopct="%1.0f%%",
        wedgeprops=dict(width=0.45, edgecolor="#0d0f14", linewidth=2),
        startangle=140,
        pctdistance=0.75
    )

    for t in texts:
        t.set_color("#e0ddd5")
        t.set_fontsize(9)

    for t in autotexts:
        t.set_color("#e0ddd5")
        t.set_fontsize(8)
        t.set_fontweight("bold")

    ax2.set_title(
        f"Units by Status\n(Total: ~{int(housing_model['units_mid'].sum())})",
        fontsize=11, fontweight="bold", pad=12
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, (ax1, ax2)

def prepare_investment_scale_data() -> pd.DataFrame:
    """
    Return the manually compiled major capital project table used in the investment chart.
    """
    inv = pd.DataFrame({
        "Project": [
            "Portage Place\nRedevelopment",
            "True North Square\n(all phases)",
            "Canadian Museum\nfor Human Rights",
            "Wehwehneh\nBahgahkinahgohn",
            "RBC Convention\nCentre Expansion",
            "Wawanesa\nHeadquarters",
            "Railside at\nThe Forks (Ph.1)",
            "300 Main\nResidential",
            "308 Colony\n(Solara Flats)",
            "St. Charles Hotel\n(est.)*",
            "Maw's Garage /\nSanford Bldg (est.)*",
            "Portage & Main\nReopening",
        ],
        "Amount_M": [650, 400, 351, 310, 180, 100, 100, 80, 77, 49, 40, 21],
        "Phase": [
            "2020+ restructuring",
            "2015–2020 transition",
            "Pre-2015 establishment",
            "2020+ restructuring",
            "Pre-2015 establishment",
            "2020+ restructuring",
            "2020+ restructuring",
            "2020+ restructuring",
            "2020+ restructuring",
            "2020+ restructuring",
            "2020+ restructuring",
            "2020+ restructuring",
        ],
        "Done": [False, True, True, False, True, True, False, True, True, False, False, True],
        "Confirmed": [True, True, True, True, True, True, False, True, True, False, False, True],
    })
    return inv


def plot_investment_scale(
    inv: pd.DataFrame,
    phase_colors: dict[str, str],
    save_path: str | None = None
):
    """
    Plot the major capital projects bar chart with confirmed vs estimated styling.
    """
    fig, ax = plt.subplots(figsize=(13, 7.5))

    for i, row in inv.iterrows():
        bar_color = phase_colors[row["Phase"]] if row["Confirmed"] else "#7a6e52"
        ax.barh(
            i, row["Amount_M"], height=0.6,
            color=bar_color,
            alpha=0.9 if row["Done"] else 0.55,
            edgecolor="none"
        )

        suffix = "" if row["Done"] else " (in progress)"
        est_tag = " [est.]" if not row["Confirmed"] else ""
        ax.text(
            row["Amount_M"] + 8, i,
            f"${row['Amount_M']}M{suffix}{est_tag}",
            va="center", fontsize=9,
            color="#e0ddd5" if row["Confirmed"] else "#8a8780"
        )

    ax.set_yticks(range(len(inv)))
    ax.set_yticklabels(inv["Project"], fontsize=9)
    ax.set_xlabel("Investment ($ Millions)")

    confirmed_total = inv[inv["Confirmed"]]["Amount_M"].sum()
    estimated_total = inv[~inv["Confirmed"]]["Amount_M"].sum()

    ax.set_title(
        f"Major Capital Projects — Confirmed: ${confirmed_total:,}M + Est.: ~${estimated_total}M [L1/L2]",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlim(0, 900)
    ax.grid(axis="x", alpha=0.3)

    ax.text(
        0.01, 0.01,
        "* St. Charles Hotel ($49M) and Maw's Garage ($40M) are cost-per-unit estimates (~$350K/unit); "
        "total project costs not confirmed in sources. Railside Ph.1 also estimated. "
        "All other values from press releases; actual expenditures may differ.",
        transform=ax.transAxes, fontsize=7, color="#8a8780", va="bottom", wrap=True
    )

    handles = [
        mpatches.Patch(color=phase_colors[p], label=p)
        for p in ["Pre-2015 establishment", "2015–2020 transition", "2020+ restructuring"]
    ]
    handles.append(
        mpatches.Patch(color="#7a6e52", alpha=0.55, label="2020+ restructuring (cost estimated)")
    )

    ax.legend(handles=handles, loc="upper right", framealpha=0.3, edgecolor="none", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax

def prepare_gantt_timeline_data(gantt: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the cleaned Gantt dataset for timeline plotting.

    Sorts events by phase, start year, and impact so the timeline reads clearly.
    """
    df_g = gantt.sort_values(
        ["phase", "year_start", "impact"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return df_g


def plot_gantt_timeline(
    df_g: pd.DataFrame,
    type_colors: dict[str, str],
    accent_color: str,
    save_path: str | None = None
):
    """
    Plot the structural event timeline with phase separators, source-quality opacity,
    and a January 2026 reference line.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    opacity_map = {"strong": 0.95, "moderate": 0.65, "weak": 0.35}
    prev_phase = None
    portage_label = "Portage Place"

    for i, row in df_g.iterrows():
        start = row["year_start"]
        dur = max(row["year_end"] - start, 0.35)
        c = type_colors.get(row["category"], "#7a7a7a")
        sq = str(row.get("source_quality", "moderate"))
        base_alpha = opacity_map.get(sq, 0.65)

        if portage_label in str(row["task_name"]) and not row["is_done"]:
            hatch = "///"
            alpha = base_alpha
        else:
            hatch = "" if row["is_done"] else "///"
            alpha = base_alpha if row["is_done"] else base_alpha * 0.6

        ax.barh(
            i, dur, left=start, height=0.6,
            color=c, alpha=alpha,
            edgecolor="#2a2e3a", linewidth=0.5, hatch=hatch
        )

        label = str(row["task_name"])
        if len(label) > 40:
            label = label[:38] + "…"

        txt_color = "#C2BCAB" if sq != "weak" else "#6a6a6a"
        ax.text(start + dur + 0.1, i, label, va="center", fontsize=7.5, color=txt_color)

        if row["phase"] != prev_phase and prev_phase is not None:
            ax.axhline(y=i - 0.5, color="#847D6E", linewidth=0.4, alpha=0.4, linestyle="--")
        prev_phase = row["phase"]

    jan_2026_x = 2026.0
    ax.axvline(x=jan_2026_x, color=accent_color, linewidth=0.8, linestyle="--", alpha=0.9)
    ax.text(
        jan_2026_x + 0.1, len(df_g) - 0.5, "Jan 2026",
        color=accent_color, fontsize=8, va="top", fontstyle="italic"
    )

    ax.set_xlim(2006, 2033)
    ax.set_xlabel("Year")
    ax.set_title(
        "Downtown Winnipeg: Structural Event Timeline [L1/L2]",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticks([])

    type_patches = [
        mpatches.Patch(color=type_colors[t], label=t)
        for t in ["Growth", "Infrastructure", "Transition", "Adaptive Reuse", "Policy"]
    ]
    prog_patch = mpatches.Patch(
        facecolor="#555", hatch="///", edgecolor="#8a8780",
        label="In progress (hatched)"
    )
    jan_line = plt.Line2D(
        [0], [0], color=accent_color, linewidth=1.5, linestyle="--",
        label="Jan 2026 reference line"
    )

    ax.legend(
        handles=type_patches + [prog_patch, jan_line],
        loc="lower right", framealpha=0.3, edgecolor="none", fontsize=8
    )

    ax.text(
        0.01, 0.01,
        "Bar opacity = source quality (full=strong, faded=weak).",
        transform=ax.transAxes, fontsize=7, color="#8a8780", va="bottom"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax

def prepare_spatial_event_data(
    filepath: str,
    x_shift: int = 120,
    y_shift: int = -20
) -> tuple[pd.DataFrame, "gpd.GeoDataFrame"]:
    """
    Load downtown project coordinates, clean them, convert to GeoDataFrame,
    and apply a small manual geometry shift for map alignment.
    """
    import geopandas as gpd

    df = pd.read_csv(filepath)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    gdf["geometry"] = gdf["geometry"].translate(xoff=x_shift, yoff=y_shift)

    return df, gdf


def plot_spatial_event_map(
    gdf,
    cat_color_map: dict[str, str],
    save_path: str | None = None
):
    """
    Plot the downtown structural events spatial map on a dark basemap.
    """
    import contextily as ctx
    import geopandas as gpd  # noqa: F401

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#0d0f14")

    for cat, grp in gdf.groupby("category"):
        grp.plot(
            ax=ax,
            markersize=60,
            color=cat_color_map.get(cat, "#7a7a7a"),
            alpha=0.9,
            edgecolor="#2a2e3a",
            linewidth=0.5,
            label=cat,
            zorder=3
        )

    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad_x = 250
    pad_y = 250
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatter,
        zoom=14
    )

    ax.set_axis_off()
    ax.set_title(
        "Downtown Winnipeg Spatial Distribution of Structural Events [L1]",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    handles = [mpatches.Patch(color=v, label=k) for k, v in cat_color_map.items()]
    ax.legend(
        handles=handles,
        framealpha=0.3,
        edgecolor="none",
        fontsize=9,
        loc="lower right"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax

def summarize_spatial_quadrants(df: pd.DataFrame) -> None:
    """
    Print east-west category counts using original longitude values.
    """
    df = df.copy()
    df["east_west"] = np.where(df["lon"] > -97.145, "East of -97.145", "West of -97.145")

    print("Projects per neighbourhood quadrant (by lon):")
    print(df.groupby(["east_west", "category"]).size().to_string())


def compute_dhs(housing: pd.DataFrame) -> dict:
    """
    Compute the Downtown Health Score (DHS) and return component scores.
    """
    housing_calc = housing[housing["include_in_model"]].copy()

    if "units_mid" not in housing_calc.columns:
        housing_calc["units_mid"] = (
            housing_calc["units_low"] + housing_calc["units_high"]
        ) / 2

    business_stability = 42

    completed = housing_calc[housing_calc["phase_status"] == "Completed"]["units_mid"].sum()
    pipeline = housing_calc[housing_calc["phase_status"] != "Completed"]["units_mid"].sum()
    built_score = min(100, completed / 6)
    pipeline_score = min(100, pipeline / 8)
    pop_score = 65

    residential_demand = int(
        0.40 * built_score
        + 0.40 * pipeline_score
        + 0.20 * pop_score
    )

    investment_pipeline = 78
    spatial_anchoring = 72
    vacancy_distress = 8

    dhs = (
        0.25 * business_stability
        + 0.25 * residential_demand
        + 0.25 * investment_pipeline
        + 0.15 * spatial_anchoring
        - 0.10 * vacancy_distress
    )
    dhs = round(dhs, 1)

    band = (
        "Stable Transition (65+)" if dhs >= 65
        else "Mixed/Transitional (50-64)" if dhs >= 50
        else "Stagnating (<50)"
    )

    return {
        "business_stability": business_stability,
        "residential_demand": residential_demand,
        "investment_pipeline": investment_pipeline,
        "spatial_anchoring": spatial_anchoring,
        "vacancy_distress": vacancy_distress,
        "dhs": dhs,
        "band": band,
        "completed_units_mid": completed,
        "pipeline_units_mid": pipeline,
        "built_score": built_score,
        "pipeline_score": pipeline_score,
        "pop_score": pop_score,
    }
def prepare_dhs_gauge_data(dhs_result: dict) -> tuple[list[tuple], float, str]:
    """
    Prepare component tuples and final band label for the DHS gauge figure.

    Parameters:
        dhs_result: Output dictionary from compute_dhs().

    Returns:
        comp_viz: List of component tuples for plotting
        dhs: Final DHS score
        band: Final band label
    """
    comp_viz = [
        ("Business\nStability", dhs_result["business_stability"], 100, 0.25, "DECLINE_C_PLACEHOLDER"),
        ("Residential\nDemand", dhs_result["residential_demand"], 100, 0.25, "GROWTH_C_PLACEHOLDER"),
        ("Investment\nPipeline", dhs_result["investment_pipeline"], 100, 0.25, "GROWTH_C_PLACEHOLDER"),
        ("Spatial\nAnchoring", dhs_result["spatial_anchoring"], 100, 0.15, "TRANS_C_PLACEHOLDER"),
        ("Vacancy &\nDistress", dhs_result["vacancy_distress"], 10, -0.10, "DECLINE_C_PLACEHOLDER"),
    ]

    return comp_viz, dhs_result["dhs"], dhs_result["band"]


def plot_dhs_gauges(
    dhs_result: dict,
    growth_color: str,
    transition_color: str,
    decline_color: str,
    accent_color: str,
    save_path: str | None = None
):
    """
    Plot the DHS component gauges and final score panel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    comp_viz = [
        ("Business\nStability", dhs_result["business_stability"], 100, 0.25, decline_color),
        ("Residential\nDemand", dhs_result["residential_demand"], 100, 0.25, growth_color),
        ("Investment\nPipeline", dhs_result["investment_pipeline"], 100, 0.25, growth_color),
        ("Spatial\nAnchoring", dhs_result["spatial_anchoring"], 100, 0.15, transition_color),
        ("Vacancy &\nDistress", dhs_result["vacancy_distress"], 10, -0.10, decline_color),
    ]

    for idx, (name, score, max_val, weight, color) in enumerate(comp_viz):
        ax = axes[idx]

        theta_bg = np.linspace(0, np.pi, 100)
        ax.plot(
            np.cos(theta_bg), np.sin(theta_bg),
            color="#2a2e3a", linewidth=14, alpha=0.4, solid_capstyle="round"
        )

        fill = score / max_val
        theta_fill = np.linspace(0, np.pi * fill, 100)
        ax.plot(
            np.cos(theta_fill), np.sin(theta_fill),
            color=color, linewidth=14, alpha=0.85, solid_capstyle="round"
        )

        prefix = "−" if weight < 0 else ""
        ax.text(0, 0.2, f"{prefix}{score}", ha="center", va="center",
                fontsize=32, fontweight="bold", color=color)
        ax.text(0, -0.15, f"/{max_val}", ha="center", va="center",
                fontsize=13, color="#8a8780")
        ax.text(0, -0.55, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#e0ddd5")
        ax.text(0, -0.85, f"{prefix}{abs(int(weight * 100))}% weight",
                ha="center", va="center", fontsize=9, color="#8a8780")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.1, 1.2)
        ax.axis("off")

    ax = axes[5]
    dhs = dhs_result["dhs"]
    band = dhs_result["band"]
    band_color = growth_color if dhs >= 65 else accent_color if dhs >= 50 else decline_color

    ax.text(0, 0.4, f"{dhs}", ha="center", va="center",
            fontsize=52, fontweight="bold", color=band_color)
    ax.text(0, -0.1, "/100", ha="center", va="center",
            fontsize=16, color="#8a8780")
    ax.text(0, -0.45, "Downtown Health Score", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#e0ddd5")
    ax.text(0, -0.75, band, ha="center", va="center",
            fontsize=10, color=band_color, fontstyle="italic")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta) * 1.05, np.sin(theta) * 1.05,
            color=band_color, linewidth=3, alpha=0.4)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.1, 1.4)
    ax.axis("off")

    fig.suptitle("DHS — Component Gauges [L4 — SCENARIO]",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, axes

def prepare_dhs_scenarios(residential_demand: int) -> tuple[dict[str, list[int]], list[float]]:
    """
    Prepare DHS sensitivity scenarios and weights.

    Parameters:
        residential_demand: Current residential demand score from DHS.

    Returns:
        scenarios: Dictionary of scenario component values
        weights: DHS component weights
    """
    scenarios = {
        "Current Assessment": [42, residential_demand, 78, 72, 8],
        "Pessimistic: Projects Delay 2+ Yrs": [35, int(residential_demand * 0.6), 55, 60, 9],
        "Optimistic: On-Time + Recovery": [58, int(residential_demand * 1.15), 85, 80, 6],
        "Stagnation: Pipeline Stalls": [30, 40, 40, 50, 9],
    }
    weights = [0.25, 0.25, 0.25, 0.15, -0.10]

    return scenarios, weights

def plot_dhs_sensitivity(
    scenarios: dict[str, list[int]],
    weights: list[float],
    growth_color: str,
    accent_color: str,
    decline_color: str,
    save_path: str | None = None
):
    """
    Plot DHS sensitivity analysis as a horizontal bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    names = list(scenarios.keys())
    scores = [round(sum(c * w for c, w in zip(v, weights)), 1) for v in scenarios.values()]
    colors = [
        growth_color if s >= 65 else accent_color if s >= 50 else decline_color if s >= 30 else "#7a3535"
        for s in scores
    ]

    bars = ax.barh(
        range(len(names)),
        scores,
        color=colors,
        alpha=0.85,
        height=0.55,
        edgecolor="none"
    )

    for i, (bar, s) in enumerate(zip(bars, scores)):
        ax.text(
            bar.get_width() + 0.8, i, f"{s}",
            va="center", fontsize=12, fontweight="bold", color="#e0ddd5"
        )

    for thresh, lbl, c in [
        (65, "Stable transition", growth_color),
        (50, "Mixed/transitional", accent_color),
        (30, "Stagnating", decline_color),
    ]:
        ax.axvline(x=thresh, color=c, linewidth=1, linestyle=":", alpha=0.5)
        ax.text(thresh + 0.5, len(names) - 0.3, lbl, fontsize=7.5, color=c, alpha=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Downtown Health Score")
    ax.set_title(
        "DHS Sensitivity Analysis: Four Scenarios [L4 — SCENARIO]",
        fontsize=14,
        fontweight="bold",
        pad=15,
        y=1.07
    )
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="#0d0f14")

    return fig, ax, scores