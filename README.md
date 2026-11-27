# DSAI4204 House Price Prediction

This course project predicts residential sale prices in Ames, Iowa, using the Kaggle House Prices dataset. The final model combines TabPFN 2.5 with 80 selected predictors and six neighborhood features. The team recorded a **Kaggle public leaderboard log RMSE of 0.10925**.

## Data and method

The dataset contains 1,460 labeled training rows and 1,459 test rows. The pipeline cleans and imputes the input fields, constructs housing features, and selects predictors using the training labels. It then adds neighborhood medians for overall quality, lot area, and year built, plus three relative measures. The final configuration, called C0, retains all training rows. TabPFN uses eight inference ensemble members and predicts centered natural-log sale prices. The pipeline restores the target mean and exponentiates the predictions for submission.

See the [data setup](data/README.md), [methodology](docs/methodology.md), and [final feature dictionary](docs/feature_dictionary.csv) for details.

## Recorded results

| Evaluation | Log RMSE |
| --- | ---: |
| Five-fold CV mean, split seed 2027 | 0.10659 |
| Five-fold CV mean, split seed 2028 | 0.10465 |
| Mean of the two CV results | 0.10562 |
| Kaggle public leaderboard | 0.10925 |

The CV runs refitted preprocessing, feature selection, and neighborhood statistics inside each training fold. Both split seeds informed model selection. The leaderboard score comes from the team's recorded public submission. [Full-precision results](results/cv_summary.csv)

## Reproduce the workflow

Run these commands from the repository root using Python 3.12. Download the [competition inputs](data/README.md) before running the pipeline.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m house_prices.reproduce features
```

The feature command writes `outputs/features/train.csv` and `outputs/features/test.csv`. The expected shape is 1,460 training rows, 1,459 test rows, and 86 predictors. The exported CSVs include row IDs, and the training export includes `SalePrice`.

Download `tabpfn-v2.5-regressor-v2.5_default.ckpt` into `checkpoints/`. The model commands verify the checkpoint SHA-256 against [the final configuration](configs/final.json). [Pinned Prior Labs checkpoint and access terms](https://huggingface.co/Prior-Labs/tabpfn_2_5/tree/6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b)

```bash
python -m house_prices.reproduce predict \
  --checkpoint checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt --device mps

python -m house_prices.reproduce evaluate \
  --checkpoint checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt --device mps
```

The prediction command writes `outputs/submission.csv` with columns `Id` and `SalePrice`. The evaluation command writes `outputs/cv_folds.csv` and `outputs/cv_summary.csv`. The original run used PyTorch MPS on Apple Silicon. Use `--device cuda` on a CUDA system or `--device cpu` with `TABPFN_ALLOW_CPU_LARGE_DATASET=1` for CPU inference. CPU runs take longer. Predictions may vary slightly across devices.

## Repository layout

| Path | Contents |
| --- | --- |
| `house_prices/` | Input loading, feature engineering, model inference, evaluation, and the command-line entry point |
| `configs/` | Final model, evaluation, and checkpoint configuration |
| `notebooks/` | Exploratory data analysis |
| `data/` | Download and input format instructions |
| `docs/` | Methodology and feature dictionary |
| `results/` | Recorded evaluation summary |

To run the [EDA notebook](notebooks/01_eda.ipynb), install JupyterLab and Seaborn in the same environment with `python -m pip install jupyterlab seaborn==0.13.2`. The notebook reads the same raw files and writes figures and tables to `outputs/eda/`.
