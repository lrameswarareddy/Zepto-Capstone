import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
    print(df.groupby("sex")["survived"].mean().round(4))
    print("\nSurvival by pclass:")
    print(df.groupby("pclass")["survived"].mean().round(4))
    print("\nSurvival by sex and pclass:")
    print(df.groupby(["sex", "pclass"])["survived"].mean().round(4))


def corr_heatmap(df):
    cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[cols].corr()
    print("\nSix-column correlation matrix:\n", corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Titanic correlation heatmap")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
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
            }
        )

    print("\nModel metrics:")
    for row in result_rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "confusion"}, indent=2))
        print("confusion_matrix:\n", row["confusion"])

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


def main():
    build_model_pipeline()


if __name__ == "__main__":
    main()
