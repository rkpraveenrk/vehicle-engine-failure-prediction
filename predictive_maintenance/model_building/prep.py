"""Make train/test splits; write to directory shared using Actions artifacts."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from features import add_features, TARGET

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='predictive_maintenance/data/engine_data.csv')
    ap.add_argument('--output', default='predictive_maintenance/data/processed')
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    if TARGET not in df.columns:
        raise ValueError(f'Missing target: {TARGET}')
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    y = pd.to_numeric(df[TARGET], errors='raise').astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError('Expected binary target values 0 and 1')
    # Keep only known raw sensors. Avoid unintended identifiers, leakage and non-numeric columns.
    from features import RAW_FEATURES
    X = add_features(df[RAW_FEATURES])
    # Missing feature imputation belongs inside the training CV pipeline, not on full dataset.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('Xtrain', Xtrain), ('Xtest', Xtest), ('ytrain', ytrain), ('ytest', ytest)]:
        data.to_csv(out / f'{name}.csv', index=False)
    print('Removed duplicate rows:', before-len(df))
    print('Feature columns:', len(X.columns), list(X.columns))
    print('Train:', Xtrain.shape, 'Test:', Xtest.shape)
    print('Prepared files saved in:', out.resolve())

if __name__ == '__main__':
    main()
