# Data setup

Download the official competition files and place them under `data/raw/` from the repository root. [Kaggle House Prices data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

| File | Role | Expected rows |
| --- | --- | ---: |
| `data/raw/train.csv` | Property predictors, `Id`, and the `SalePrice` target | 1,460 |
| `data/raw/test.csv` | Property predictors and `Id` in competition submission order | 1,459 |

The raw files contain 79 predictors. During CSV parsing, blank fields and `NA` represent missing values. Literal categories such as `None` remain available to the preprocessing rules. The competition page also provides `data_description.txt` and `sample_submission.csv` as references.

The evaluation command uses the training CSV. Feature export, prediction, and the exploratory notebook use both CSVs. Keep downloaded inputs in `data/raw/` and generated files under `outputs/`. Git ignores both locations.
