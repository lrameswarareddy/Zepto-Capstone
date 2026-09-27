# Analytics Pipeline

Run from the repository root:

```powershell
python analytics\analytics_pipeline.py
```

The script reads `titanic.csv` after the first dataset creation, so later runs are offline. It reports `df.info()`, `df.describe()`, shape, missing percentages, classifier metrics, imbalance comparison, Random Forest tuning, and fare-regression metrics.

## Cleaning Decisions

- `embarked` has 0.22% missing values, so rows are dropped under the under-5% rule.
- `age` has 19.87% missing values, so it is median-imputed under the 5%-30% rule.
- `deck` has 77.22% missing values and is dropped because reliable imputation is not defensible.
- The modeling `ColumnTransformer` independently imputes, one-hot encodes, and scales only after the stratified train/test split.

## EDA Interpretations

- The age histogram shows a broad adult-age distribution with a smaller upper tail; it helps establish the shape and concentration of passenger ages. The age box plot reports 65 IQR outliers, which are retained because they represent plausible passenger ages and the model pipeline is robust to them.
- The fare histogram shows a strongly right-skewed distribution: the mean fare is 32.10, above the median 14.45, which is above the mode 8.05. The fare box plot reports 114 IQR outliers, confirming that a small number of expensive tickets drive a long upper tail.
- The survival-by-sex/class chart shows that female passengers survived more often than male passengers in every class. The strongest contrast is between first-class females (96.74%) and third-class males (13.54%), indicating that sex and class jointly explain meaningful variation.
- The six-column heatmap excludes the redundant boolean flags and uses exactly survived, pclass, age, sibsp, parch, and fare. The two strongest absolute correlations are pclass-fare (-0.548) and sibsp-parch (0.415); class is associated with ticket price, while family group variables move together.
- Z-score summaries for age and fare have means approximately 0 and standard deviations approximately 1, confirming the exploratory standardization step.
- The ROC chart compares all three classifiers against the chance diagonal; Logistic Regression has the highest AUC at 0.8610, while the other curves show useful but weaker ranking performance.
- The residual plot shows a wider residual spread for higher fitted fares: the lower/upper fitted-half standard deviations are 13.2160 and 57.2069. This is evidence of heteroscedasticity rather than constant residual variance.
- The decision-tree chart is limited to four displayed levels so the one-hot encoded feature labels remain readable; it highlights the strongest early splits while the fitted tree remains unchanged.

## Modeling Results

The identical stratified 80/20 split is used for all classifiers. Stratification preserves the observed approximately 61.7% non-survived and 38.3% survived class proportions in both folds.

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7697 | 0.6901 | 0.7206 | 0.7050 | 0.7541 |
| Random Forest | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |

Imbalance comparison on the test fold:

| Variant | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.7937 | 0.7353 | 0.7634 |
| `class_weight="balanced"` | 0.7429 | 0.7647 | 0.7536 |
| SMOTE on training fold only | 0.7385 | 0.7059 | 0.7218 |

The baseline has the best F1 and precision, while balanced weights improve recall. SMOTE is applied only to transformed training data and performs worst on this split, so the baseline Random Forest is preferred for balanced overall performance.

Grid search tunes `n_estimators`, `max_depth`, `min_samples_split`, and `max_features`. The validated run selects `n_estimators=50`, `max_depth=5`, `min_samples_split=5`, and `max_features="sqrt"`; the fitted estimator reports an OOB score of 0.8158.

The fare regression reports MAE 21.0986, RMSE 41.7021, R2 0.3482, and adjusted R2 0.3091. The regression is useful as a baseline but leaves substantial fare variation unexplained, and the measured residual spread confirms heteroscedasticity. Random Forest is the recommended classifier because it has the strongest accuracy (0.8202) and F1 (0.7576), while Logistic Regression has the strongest AUC (0.8610); the deployment choice depends on whether balanced classification or ranking quality is more important.

Generated artifacts include `age_hist.png`, `age_box.png`, `fare_hist.png`, `fare_box.png`, `survival_by_sex_class.png`, `correlation_heatmap.png`, `roc_curves.png`, `fare_regression_residuals.png`, `decision_tree.png`, and `best_pipeline.joblib`.
