import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    """
    Loads and cleans data from an Excel file following strict cleaning rules.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    
    df_seap1 = sheets.get("SEA P1")
    df_seap2 = sheets.get("SEA P2")
    df_progd = sheets.get("Programmatic Display")
    df_partner = sheets.get("Partner")

    # Clean SEA P1
    df_seap1 = df_seap1.drop(index=[0, 1])
    df_seap1.columns = df_seap1.iloc[0]
    df_seap1 = df_seap1.iloc[1:].reset_index(drop=True)
    df_seap1 = df_seap1.round().astype(int)

    # Clean SEA P2
    df_seap2 = df_seap2.drop(index=[0, 1])
    df_seap2.columns = df_seap2.iloc[0]
    df_seap2 = df_seap2.iloc[1:].reset_index(drop=True)
    df_seap2 = df_seap2.round().astype(int)

    # Clean Programmatic Display
    df_progd.columns = df_progd.iloc[0]
    df_progd = df_progd.iloc[1:].reset_index(drop=True)
    df_progd = df_progd.rename(columns={"Impr.": "Impressions", "Clicks ": "Clicks"})
    
    # Convert numeric columns
    numeric_columns = ['Costs', 'Applications']
    for col in numeric_columns:
        df_progd[col] = pd.to_numeric(df_progd[col], errors='coerce')
    df_progd = df_progd.dropna(subset=numeric_columns)
    
    return {
        "SEA P1": df_seap1,
        "SEA P2": df_seap2,
        "Programmatic Display": df_progd,
        "Partner": df_partner
    }

def fit_polynomial_models(df_seap1, df_seap2):
    """
    Fits polynomial models to SEA P1 and SEA P2 data.
    """
    x_p1 = df_seap1['Cost'].values
    y_p1 = df_seap1['Applications'].values
    poly_p1 = np.poly1d(np.polyfit(x_p1, y_p1, 2))
    
    x_p2 = df_seap2['Cost'].values
    y_p2 = df_seap2['Applications'].values
    poly_p2 = np.poly1d(np.polyfit(x_p2, y_p2, 2))
    
    return poly_p1, poly_p2