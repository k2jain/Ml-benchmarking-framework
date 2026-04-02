import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class BenchmarkResult:
    model_name: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    latency_ms: float
    pr_auc: float
    conf_matrix: np.ndarray
    best_params: Dict[str, Any]


def load_data():
    """
    Loads a clean binary classification dataset for benchmarking.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def evaluate_model(model, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """
    Fits a model, measures prediction latency, and returns evaluation metrics.
    """
    model.fit(X_train, y_train)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "latency_ms": latency_ms,
        "pr_auc": average_precision_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred),
    }


def optimize_logistic_regression(X_train, y_train, n_trials=20):
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, solver=solver, max_iter=1000)),
        ])

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_random_forest(X_train, y_train, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42,
        }

        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_xgboost(X_train, y_train, n_trials=20):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_mlp(X_train, y_train, n_trials=20):
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=500,
                random_state=42,
            )),
        ])

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def build_models(best_lr, best_rf, best_xgb, best_mlp):
    models = []

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_lr["C"],
            solver=best_lr["solver"],
            max_iter=1000,
        )),
    ])
    models.append(("Logistic Regression", lr_model, best_lr))

    rf_model = RandomForestClassifier(
        n_estimators=best_rf["n_estimators"],
        max_depth=best_rf["max_depth"],
        min_samples_split=best_rf["min_samples_split"],
        random_state=42,
    )
    models.append(("Random Forest", rf_model, best_rf))

    xgb_model = XGBClassifier(
        n_estimators=best_xgb["n_estimators"],
        max_depth=best_xgb["max_depth"],
        learning_rate=best_xgb["learning_rate"],
        subsample=best_xgb["subsample"],
        colsample_bytree=best_xgb["colsample_bytree"],
        eval_metric="logloss",
        random_state=42,
    )
    models.append(("XGBoost", xgb_model, best_xgb))

    mlp_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(best_mlp["hidden_size"],),
            alpha=best_mlp["alpha"],
            learning_rate_init=best_mlp["learning_rate_init"],
            max_iter=500,
            random_state=42,
        )),
    ])
    models.append(("Neural Network", mlp_model, best_mlp))

    return models


def save_results(results: List[BenchmarkResult], output_csv="outputs/benchmark_results.csv"):
    df = pd.DataFrame([
        {
            "model_name": r.model_name,
            "accuracy": r.accuracy,
            "f1": r.f1,
            "precision": r.precision,
            "recall": r.recall,
            "latency_ms": r.latency_ms,
            "pr_auc": r.pr_auc,
            "best_params": str(r.best_params),
        }
        for r in results
    ])

    df = df.sort_values(by="f1", ascending=False)
    df.to_csv(output_csv, index=False)
    return df


def plot_results(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.bar(df["model_name"], df["f1"])
    plt.title("Model Comparison by F1 Score")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/f1_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_name"], df["latency_ms"])
    plt.title("Inference Latency by Model")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/latency_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_name"], df["pr_auc"])
    plt.title("Model Comparison by PR-AUC")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/pr_auc_comparison.png")
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    print("Optimizing Logistic Regression...")
    best_lr = optimize_logistic_regression(X_train, y_train)

    print("Optimizing Random Forest...")
    best_rf = optimize_random_forest(X_train, y_train)

    print("Optimizing XGBoost...")
    best_xgb = optimize_xgboost(X_train, y_train)

    print("Optimizing Neural Network...")
    best_mlp = optimize_mlp(X_train, y_train)

    models = build_models(best_lr, best_rf, best_xgb, best_mlp)

    results = []
    for model_name, model, params in models:
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        result = BenchmarkResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            latency_ms=metrics["latency_ms"],
            pr_auc=metrics["pr_auc"],
            conf_matrix=metrics["conf_matrix"],
            best_params=params,
        )
        results.append(result)

    df = save_results(results)
    plot_results(df)

    print("\n=== FINAL LEADERBOARD ===")
    print(df.to_string(index=False))

    print("\n=== CONFUSION MATRICES ===")
    for r in results:
        print(f"\n{r.model_name}")
        print(r.conf_matrix)

    print("\nSaved:")
    print("- outputs/benchmark_results.csv")
    print("- outputs/f1_comparison.png")
    print("- outputs/latency_comparison.png")
    print("- outputs/pr_auc_comparison.png")


if __name__ == "__main__":
    main()
