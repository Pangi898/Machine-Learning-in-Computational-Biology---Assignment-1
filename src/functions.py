# Prerequisites

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

RANDOM_STATE = 42

# Data loading & splitting

def load_data(dev_path: str, eval_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dev_df = pd.read_csv(dev_path, index_col=0)
    eval_df = pd.read_csv(eval_path, index_col=0)
    return dev_df, eval_df


def train_val_split(
    dev_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    age = dev_df["age"]
    age_bins = pd.qcut(age, q=10, duplicates="drop")
    train_idx, val_idx = train_test_split(
        dev_df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=age_bins
    )
    return dev_df.loc[train_idx], dev_df.loc[val_idx]

def summarize_split(df, name):
    print(f"=== {name} ===")
    print("Samples:", df.shape[0])
    print("Features:", df.shape[1])
    print(f"Age mean ± std: {df['age'].mean():.2f} ± {df['age'].std():.2f}")
    print(f"Age range: {df['age'].min()} – {df['age'].max()}")
    print("\nSex distribution:")
    print(df["sex"].value_counts(dropna=False))
    print("\nEthnicity distribution:")
    print(df["ethnicity"].value_counts(dropna=False))
    print("\n")

# Feature matrices

def get_X_y(
    df: pd.DataFrame,
    use_cpg: bool = True,
    use_metadata: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = df["age"]  # keep as Series for consistency
    cpg_cols = [c for c in df.columns if c.startswith("cg")]
    meta_cols = ["sex", "ethnicity"]

    feature_cols: List[str] = []
    if use_cpg:
        feature_cols += cpg_cols
    if use_metadata:
        feature_cols += meta_cols

    X = df[feature_cols].copy()
    return X, y, feature_cols

# Preprocessing

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    use_cpg: bool = True,
    use_metadata: bool = True
) -> Tuple[Pipeline, List[str], List[str]]:
    categorical_cols = ["sex", "ethnicity"]
    cpg_cols = [c for c in df.columns if c.startswith("cg")]

    feature_cols: List[str] = []
    if use_cpg:
        feature_cols += cpg_cols
    if use_metadata:
        feature_cols += categorical_cols

    numeric_features = [c for c in feature_cols if c in cpg_cols]
    categorical_features = [c for c in feature_cols if c in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    return pipeline, numeric_features, categorical_features

# Model pipelines

def build_model_pipeline(preprocess_pipeline: Pipeline, model) -> Pipeline:
    # keep the full preprocessing pipeline, not just its inner step
    return Pipeline([
        ("preprocess", preprocess_pipeline),
        ("model", model)
    ])

# Bootstrap utilities

def bootstrap_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rmse_vals = []
    r2_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]

        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))
        r2_vals.append(r2_score(yt, yp))

    return np.array(rmse_vals), np.array(r2_vals)


def bootstrap_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)

    rmse_vals = []
    mae_vals = []
    r2_vals = []
    r_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]

        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))
        mae_vals.append(mean_absolute_error(yt, yp))
        r2_vals.append(r2_score(yt, yp))
        r_vals.append(pearsonr(yt, yp)[0])

    def ci(arr):
        return np.percentile(arr, [2.5, 97.5])

    return {
        "RMSE": (np.mean(rmse_vals), *ci(rmse_vals)),
        "MAE": (np.mean(mae_vals), *ci(mae_vals)),
        "R2": (np.mean(r2_vals), *ci(r2_vals)),
        "R": (np.mean(r_vals), *ci(r_vals)),
    }


def bootstrap_eval(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n: int = 1000
) -> Dict[str, Tuple[float, float, float]]:
    # ensure y is array-like
    y = np.asarray(y)
    preds = model.predict(X)

    metrics = {
        "rmse": [],
        "mae": [],
        "r2": [],
        "r": []
    }

    for _ in range(n):
        idx = resample(range(len(y)), replace=True)
        idx = np.array(idx)
        y_b = y[idx]
        p_b = preds[idx]

        metrics["rmse"].append(np.sqrt(mean_squared_error(y_b, p_b)))
        metrics["mae"].append(mean_absolute_error(y_b, p_b))
        metrics["r2"].append(r2_score(y_b, p_b))
        metrics["r"].append(np.corrcoef(y_b, p_b)[0, 1])

    def summarize(arr):
        return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    return {
        "RMSE": summarize(metrics["rmse"]),
        "MAE": summarize(metrics["mae"]),
        "R2": summarize(metrics["r2"]),
        "R": summarize(metrics["r"]),
    }

# Feature selection helpers

def mrmr_importance_df(selected_features: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": selected_features,
        "importance_rank": range(1, len(selected_features) + 1)
    })


def evaluate_feature_set(train_df, val_df, feature_list):
    # Build preprocessing only for the selected CpGs
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_list)
        ]
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", BayesianRidge())
    ])

    X_train_fs = train_df[feature_list]
    X_val_fs = val_df[feature_list]

    pipeline.fit(X_train_fs, train_df["age"])
    preds = pipeline.predict(X_val_fs)

    return bootstrap_metrics(val_df["age"], preds)

# Hyperparameter tuning

def tune_model(
    pipe: Pipeline,
    params: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = RANDOM_STATE
) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X, y)
    return search  # use search.best_estimator_ downstream

# Table builder

def make_table(name: str, stage: str, results: Dict) -> pd.DataFrame:
    rm, lo_rm, hi_rm = results["RMSE"]
    mae, _, _ = results["MAE"]
    r2, _, _ = results["R2"]
    r, _, _ = results["R"]

    return pd.DataFrame([[
        name, stage,
        rm, f"[{lo_rm:.2f}, {hi_rm:.2f}]",
        mae, r2, r
    ]], columns=["Model", "Stage", "RMSE (mean)", "95% CI", "MAE", "R2", "Pearson r"])