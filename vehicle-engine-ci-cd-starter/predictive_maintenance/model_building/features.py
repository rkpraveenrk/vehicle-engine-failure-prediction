"""Shared feature engineering used during training and live prediction."""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
    "lub oil temp", "Coolant temp",
]
TARGET = "Engine Condition"

def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [x for x in RAW_FEATURES if x not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw sensor columns: {missing}")
    df = raw.copy()
    for name in RAW_FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["delta_temp"] = df["Coolant temp"] - df["lub oil temp"]
    df["oil_viscosity_proxy"] = df["Lub oil pressure"] / (df["lub oil temp"] + 1e-5)
    df["cooling_eff_index"] = df["Coolant pressure"] / (df["Coolant temp"] + 1e-5)
    df["oil_temp_squared"] = df["lub oil temp"] ** 2
    df["coolant_temp_squared"] = df["Coolant temp"] ** 2
    df["thermal_ratio"] = df["lub oil temp"] / (df["Coolant temp"] + 1e-5)
    df["fuel_eff_proxy"] = df["Fuel pressure"] / (df["Engine rpm"] + 1e-5)
    df["lube_stress"] = df["Engine rpm"] / (df["Lub oil pressure"] + 1e-5)
    df["press_balance"] = df["Lub oil pressure"] / (df["Fuel pressure"] + 1e-5)
    df["cool_vs_lube_balance"] = df["Coolant pressure"] / (df["Lub oil pressure"] + 1e-5)
    df["total_pressure"] = df["Lub oil pressure"] + df["Fuel pressure"] + df["Coolant pressure"]
    df["rpm_x_oilpress"] = df["Engine rpm"] * df["Lub oil pressure"]
    df["temp_x_fuelpress"] = df["Coolant temp"] * df["Fuel pressure"]
    df["rpm_x_temp"] = df["Engine rpm"] * df["Coolant temp"]
    df["oilpress_x_fuelpress"] = df["Lub oil pressure"] * df["Fuel pressure"]
    return df.replace([np.inf, -np.inf], np.nan)
