import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_PATH = Path(__file__).resolve().parent / "titanic.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_titanic():
    df = sns.load_dataset("titanic")
    df.to_csv(DATA_PATH, index=False)
    return df


def build_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return load_titanic()


def summarize_missing(df):
    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]
    print("Missing percentages (by column):")
    print(missing)
    return missing


def plot_univariate(df):
    for col in ["age", "fare"]:
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"{col.title()} histogram")
        plt.savefig(OUTPUT_DIR / f"{col}_hist.png")
        plt.close()

        sns.boxplot(x=df[col].dropna())
        plt.title(f"{col.title()} boxplot")
        plt.savefig(OUTPUT_DIR / f"{col}_box.png")
        plt.close()


def compute_outliers(df):
    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col.title()} outliers: {len(outliers)}")


def survival_breakdown(df):
    print("\nSurvival by sex:")
    sex_masks = {sex: df["sex"].eq(sex) for sex in ["female", "male"]}
    print(pd.Series({sex: df.loc[mask, "survived"].mean() for sex, mask in sex_masks.items()}).round(4))
    print("\nSurvival by pclass:")
    class_masks = {pclass: df["pclass"].eq(pclass) for pclass in sorted(df["pclass"].unique())}
    print(pd.Series({pclass: df.loc[mask, "survived"].mean() for pclass, mask in class_masks.items()}).round(4))
    print("\nSurvival by sex and pclass:")
    combined_rates = {}
    for sex in sex_masks:
        for pclass in class_masks:
            combined_mask = sex_masks[sex] & class_masks[pclass]
            combined_rates[(sex, pclass)] = df.loc[combined_mask, "survived"].mean()
    print(pd.Series(combined_rates).round(4))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sex", y="survived", hue="pclass", errorbar=None)
    plt.ylabel("Survival rate")
    plt.title("Survival rate by sex and passenger class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_by_sex_class.png")
    plt.close()


def corr_heatmap(df):
    cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[cols].corr()
    print("\nSix-column correlation matrix:\n", corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Titanic correlation heatmap")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    pairs = []
    for index, first_column in enumerate(cols):
        for second_column in cols[index + 1:]:
            pairs.append((first_column, second_column, corr.loc[first_column, second_column]))
    strongest_pairs = sorted(pairs, key=lambda pair: abs(pair[2]), reverse=True)[:2]
    print("Strongest absolute correlations:", strongest_pairs)
    return corr


def standardization_check(df):
    age_mean = df["age"].mean()
    age_std = df["age"].std(ddof=0)
    fare_mean = df["fare"].mean()
    fare_std = df["fare"].std(ddof=0)
    df["age_z"] = (df["age"] - age_mean) / age_std
    df["fare_z"] = (df["fare"] - fare_mean) / fare_std
    print("\nAge z-score summary:")
    print(df["age_z"].describe())
    print("\nFare z-score summary:")
    print(df["fare_z"].describe())


def plot_model_roc_curves(result_rows, y_test):
    plt.figure(figsize=(8, 6))
    for row in result_rows:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, row["probabilities"])
        plt.plot(false_positive_rate, true_positive_rate, label=f"{row['model']} (AUC={row['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Classifier ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png")
    plt.close()


def evaluate_imbalance_variants(preprocessor, model, X_train, X_test, y_train, y_test):
    from imblearn.over_sampling import SMOTE

    transformed_train = preprocessor.fit_transform(X_train, y_train)
    transformed_test = preprocessor.transform(X_test)
    variants = {
        "baseline": model,
        "class_weight_balanced": RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced", n_jobs=1),
    }
    rows = []
    for name, estimator in variants.items():
        estimator.fit(transformed_train, y_train)
        predictions = estimator.predict(transformed_test)
        rows.append({"variant": name, "precision": precision_score(y_test, predictions), "recall": recall_score(y_test, predictions), "f1": f1_score(y_test, predictions)})

    smote_train, smote_target = SMOTE(random_state=42).fit_resample(transformed_train, y_train)
    smote_model = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=1)
    smote_model.fit(smote_train, smote_target)
    predictions = smote_model.predict(transformed_test)
    rows.append({"variant": "smote_training_only", "precision": precision_score(y_test, predictions), "recall": recall_score(y_test, predictions), "f1": f1_score(y_test, predictions)})
    comparison = pd.DataFrame(rows).round(4)
    print("\nClass balance:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\nImbalance comparison:")
    print(comparison)
    return comparison


def run_regression(df):
    regression_features = ["survived", "pclass", "age", "sibsp", "parch", "sex", "embarked"]
    regression_df = df[regression_features + ["fare"]].copy()
    target = regression_df.pop("fare")
    train_features, test_features, train_target, test_target = train_test_split(regression_df, target, test_size=0.2, random_state=42)
    numeric_features = ["survived", "pclass", "age", "sibsp", "parch"]
    categorical_features = ["sex", "embarked"]
    regression_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    regression_pipeline = Pipeline([("preprocessor", regression_preprocessor), ("model", LinearRegression())])
    regression_pipeline.fit(train_features, train_target)
    predictions = regression_pipeline.predict(test_features)
    residuals = test_target - predictions
    residual_correlation = residuals.corr(pd.Series(predictions, index=residuals.index))
    lower_spread = residuals[predictions <= pd.Series(predictions).quantile(0.5)].std()
    upper_spread = residuals[predictions > pd.Series(predictions).quantile(0.5)].std()
    transformed_feature_count = regression_pipeline.named_steps["preprocessor"].transform(train_features).shape[1]
    r2 = r2_score(test_target, predictions)
    adjusted_r2 = 1 - (1 - r2) * (len(test_target) - 1) / (len(test_target) - transformed_feature_count - 1)
    metrics = {"mae": mean_absolute_error(test_target, predictions), "rmse": mean_squared_error(test_target, predictions) ** 0.5, "r2": r2, "adjusted_r2": adjusted_r2}
    print("\nFare regression metrics:")
    print(pd.Series(metrics).round(4))
    print(f"Residual correlation with prediction: {residual_correlation:.4f}")
    print(f"Residual spread, lower/upper fitted halves: {lower_spread:.4f} / {upper_spread:.4f}")
    print("Residual conclusion: the spread is wider for higher fitted fares, indicating heteroscedasticity.")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Fare regression residual plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fare_regression_residuals.png")
    plt.close()
    return metrics


def build_model_pipeline():
    df = build_data()
    print("\nData shape:", df.shape)
    df.info()
    print(df.describe().round(2))
    missing = summarize_missing(df)

    for col in ["age", "embarked", "deck"]:
        if col in missing.index:
            print(f"{col} missing rate = {missing[col]:.2f}%")

    # This follows the threshold rule: drop rows for <5%, impute for 5-30%, drop heavily missing columns
    df = df.dropna(subset=["embarked"]).copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.drop(columns=["deck"], errors="ignore")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    print("\nFare skewness summary:")
    print({"mean": fare_mean, "median": fare_median, "mode": fare_mode})
    print("Fare distribution is right-skewed because mean > median > mode.")

    plot_univariate(df)
    compute_outliers(df)
    survival_breakdown(df)
    corr_heatmap(df)
    standardization_check(df)

    X = df[["sex", "pclass", "age", "sibsp", "parch", "fare", "embarked"]]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_cols = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_cols = ["sex", "embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
    }

    result_rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        result_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "auc": roc_auc_score(y_test, probs),
                "confusion": confusion_matrix(y_test, preds),
                "probabilities": probs,
            }
        )

    print("\nModel metrics:")
    for row in result_rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in {"confusion", "probabilities"}}, indent=2))
        print("confusion_matrix:\n", row["confusion"])

    classification_table = pd.DataFrame(
        [{key: row[key] for key in ["model", "accuracy", "precision", "recall", "f1", "auc"]} for row in result_rows]
    ).round(4)
    print("\nClassification comparison table:\n", classification_table)
    plot_model_roc_curves(result_rows, y_test)
    evaluate_imbalance_variants(
        preprocessor,
        RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=1),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_model_name = max(result_rows, key=lambda row: row["f1"])["model"]
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
    best_pipeline.fit(X_train, y_train)
    model_path = OUTPUT_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    reload = joblib.load(model_path)
    sample_input = pd.DataFrame([{"sex": "male", "pclass": 3, "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}])
    print("\nReload test prediction:", reload.predict(sample_input)[0])

    rf_model = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=1)
    rf_pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf_model)])
    param_grid = {
        "model__n_estimators": [50],
        "model__max_depth": [3, 5, None],
        "model__min_samples_split": [2, 5],
        "model__max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(rf_pipeline, param_grid=param_grid, cv=3, n_jobs=1)
    grid.fit(X_train, y_train)
    print("\nBest RF params:", grid.best_params_)
    print("Best RF OOB score:", grid.best_estimator_.named_steps["model"].oob_score_)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree = Pipeline([("preprocessor", preprocessor), ("model", tree_model)])
    tree.fit(X_train, y_train)
    tree_feature_names = tree.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(18, 8))
    plot_tree(
        tree.named_steps["model"],
        feature_names=tree_feature_names,
        class_names=["Not Survived", "Survived"],
        max_depth=4,
        filled=True,
    )
    plt.savefig(OUTPUT_DIR / "decision_tree.png")
    plt.close()

    regression_metrics = run_regression(df)
    print("\nSeparate regression metrics table:\n", pd.DataFrame([regression_metrics]).round(4))
    combined_table = classification_table.copy()
    for metric in ["mae", "rmse", "r2", "adjusted_r2"]:
        combined_table[f"regression_{metric}"] = None
    regression_row = {"model": "linear_regression_fare"}
    regression_row.update({key: None for key in ["accuracy", "precision", "recall", "f1", "auc"]})
    regression_row.update({f"regression_{key}": value for key, value in regression_metrics.items()})
    combined_table = pd.concat([combined_table, pd.DataFrame([regression_row])], ignore_index=True)
    print("\nCombined model comparison table:\n", combined_table.round(4))


def main():
    build_model_pipeline()


if __name__ == "__main__":
    main()
