# Methodology

## Preprocessing and feature selection

The final C0 representation applies domain-specific rules for missing amenities and inconsistent source fields, builds derived housing attributes, and uses training statistics to impute the remaining missing values. Derived features include area totals, age, quality interactions, and ratios in the original measurement units. The pipeline applies `log1p` to selected skewed nonnegative area and value fields. `MSSubClass` is categorical.

The feature selector ranks predictors using mutual information, random forest feature importance, and grouped permutation importance from a Ridge model. It estimates the ranking across three inner training splits, filters constant and weak near-constant fields, and avoids retaining highly redundant pairs. It selects 80 predictors. Six neighborhood features add training medians for `OverallQual`, `LotArea`, and `YearBuilt`, ratios to the first two medians, and a difference from the third. Unseen neighborhoods use the corresponding global training medians.

The [feature dictionary](feature_dictionary.csv) describes the 86 columns from the full training fit in export order. Features selected inside individual CV folds may differ.

## Model and evaluation

TabPFN 2.5 uses eight ensemble members, seed 42, and float32 inference with the pinned regression checkpoint. For each training fit, the model uses the natural logarithm of `SalePrice` after subtracting that fit's target mean. Log predictions are shifted back before evaluation and exponentiated for the Kaggle export.

The evaluation uses shuffled five-fold cross-validation with split seeds 2027 and 2028. Every validation fold is passed through transformations fitted from that fold's training rows. Scores are RMSE on natural-log sale prices, with a mean across five folds for each seed. Both split seeds informed the post-hoc selection of the submitted configuration.

For the final submission, the pipeline refits on all 1,460 labeled rows, retains all 1,459 test rows, and preserves their original IDs and order. The recorded Kaggle public score and the original submission checksum are stored in [the final configuration](../configs/final.json).
