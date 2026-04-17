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


def assign_phase(year: int) -> str:
    """Assign structural phase from year."""
    if year < 2015:
        return "Pre-2015 establishment"
    elif year <= 2020:
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

    Encodes confidence and source quality as numeric features, defines the
    feature set and target column, and drops rows with missing values.

    Parameters:
        df: Input housing DataFrame.

    Returns:
        A tuple containing:
        - prepared prediction DataFrame
        - list of feature column names
        - target column name
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

