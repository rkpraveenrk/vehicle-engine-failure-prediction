"""Dataset registration and structural validation (rubric: Data Registration)."""
import argparse
from pathlib import Path
import pandas as pd
from features import RAW_FEATURES, TARGET

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='predictive_maintenance/data/engine_data.csv')
    args = parser.parse_args()
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path.resolve()}. Upload engine_data.csv to data/.")
    df = pd.read_csv(path)
    expected = RAW_FEATURES + [TARGET]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    if df.empty:
        raise ValueError('Dataset is empty')
    labels = pd.to_numeric(df[TARGET], errors='coerce')
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("Engine Condition must be binary containing 0 and 1 without nulls")
    if any(pd.to_numeric(df[c], errors='coerce').notna().sum() == 0 for c in RAW_FEATURES):
        raise ValueError('A required sensor column contains no numeric values')
    print('DATA REGISTRATION VALIDATION PASSED')
    print('Path:', path.resolve())
    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Duplicates:', int(df.duplicated().sum()))
    print('Missing values per column:\n', df.isna().sum().to_string())
    print('Target distribution:\n', labels.value_counts().sort_index().to_string())

if __name__ == '__main__':
    main()
