# COMP - 2040: Final Project  
# **Name:** Troy Dela Rosa  
# **SID#** 0213352  
# **Date** April 10, 2026  
# **Instructor:** Chris Mac  
# **WEEK 13:** Source file


import pandas as pd

def clean_column_names(df):
    """
    Standardize column names by converting to lowercase and replacing spaces with underscores.

    This improves consistency and makes column referencing easier during analysis.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def create_is_closed(df):
    """
    Create a binary column 'is_closed' to identify inactive businesses.

    A business is considered closed if its status is:
    'Closed (L)', 'Ceased Operation', or 'Cancelled'.

    This serves as a proxy for business closure since no explicit field exists.
    """
    closed_status = ["Closed (L)", "Ceased Operation", "Cancelled"]
    df["is_closed"] = df["status"].isin(closed_status)
    return df


def filter_downtown(df):
    """
    Filter dataset to include only downtown records.

    Uses the 'community_characterization_area' field to isolate
    businesses located in downtown Winnipeg.
    """
    return df[df["community_characterization_area"] == "Downtown"]
