"""Use the saved final pipeline(s); no fitting or validation occurs here."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT.parent/'test.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results/submission_reproduced.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.input, keep_default_na=False, na_values=['NA', ''])
    artifact = joblib.load(ROOT/'models/best_model.joblib')
    pred = np.average([pipe.predict(data.drop(columns='Id')) for _, pipe in artifact['models']],
                      axis=0, weights=artifact['weights'])
    out = pd.DataFrame({'Id': data['Id'], 'SalePrice': np.exp(pred)})
    assert out['SalePrice'].gt(0).all() and np.isfinite(out['SalePrice']).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'Saved {len(out)} predictions to {args.output.resolve()}')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
