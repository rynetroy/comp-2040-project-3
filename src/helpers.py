# COMP - 2040: Final Project  
# **Name:** Troy Dela Rosa  
# **SID#** 0213352  
# **Date** April 10, 2026  
# **Instructor:** Chris Mac  
# **WEEK 13:** Source file


import pandas as pd

# Replace spaces with underscores
def clean_column_names(df):
    """Standardize column names."""
    df.columns = df.columns.str.lower().str.replace(" ", "_") 
    return df

# Create a new column 'is_closed' that is True
# if 'status' is in the closed_status list, otherwise False
def create_is_closed(df):
    """Create closed business indicator."""
    closed_status = ["Closed (L)", "Ceased Operation", "Cancelled"]
    df["is_closed"] = df["status"].isin(closed_status)
    return df

def filter_downtown(df):
    """Filter to downtown records."""
    return df[df["community_characterization_area"] == "Downtown"]