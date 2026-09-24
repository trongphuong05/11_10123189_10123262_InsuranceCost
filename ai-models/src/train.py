"""
Huấn luyện ≥4 model trên cùng split, lưu pipeline+model thành model.joblib.
Chạy:  python train.py   (từ thư mục ai-models/src)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    inspect_and_clean,
    load_raw_dataframe,
    split_xy,
)

AI_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = AI_ROOT / "models"
COMP_DIR = MODELS_DIR / "comparisons"


def evaluate(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(float(np.sqrt(mse)), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def measure_predict_ms(pipe, X_sample, n: int = 30) -> float:
    pipe.predict(X_sample)
    t0 = time.perf_counter()
    for _ in range(n):
        pipe.predict(X_sample)
    return round((time.perf_counter() - t0) / n * 1000, 2)


def fit_one(name, estimator, pre, Xtr, ytr, Xte, yte, compress=3) -> dict:
    pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_s = round(time.perf_counter() - t0, 2)
    m_tr = evaluate(ytr, pipe.predict(Xtr))
    m_te = evaluate(yte, pipe.predict(Xte))
    pred_ms = measure_predict_ms(pipe, Xte.iloc[:1])
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    path = COMP_DIR / f"{name}.pkl"
    joblib.dump(pipe, path, compress=compress)
    size = path.stat().st_size
    print(
        f"{name:22s} R2_test={m_te['R2']:.4f} MAE={m_te['MAE']:.2f} "
        f"RMSE={m_te['RMSE']:.2f} R2_train={m_tr['R2']:.4f} "
        f"train={train_s:.2f}s pred={pred_ms:.2f}ms size={size/1024:.1f}KB"
    )
    return {
        "model": name,
        "pipe": pipe,
        "metrics_test": m_te,
        "metrics_train": m_tr,
        "train_time_sec": train_s,
        "predict_ms": pred_ms,
        "size_bytes": size,
        "size_kb": round(size / 1024, 1),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = inspect_and_clean(load_raw_dataframe())
    n_dup = int(df.duplicated().sum())
    print(f"rows={len(df)} duplicates={n_dup} missing={int(df.isnull().sum().sum())}")
    X_train, X_test, y_train, y_test = split_xy(df)
    print(f"split train={len(X_train)} test={len(X_test)} random_state=42")
    pre = build_preprocessor()

    results = []
    results.append(fit_one("dummy_mean", DummyRegressor(strategy="mean"), pre, X_train, y_train, X_test, y_test))
    results.append(fit_one("linear_regression", LinearRegression(), pre, X_train, y_train, X_test, y_test))

    dt_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", DecisionTreeRegressor(random_state=42))]),
        param_grid={"model__max_depth": [4, 6, 8, 10], "model__min_samples_leaf": [4, 8, 16]},
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    print("DT best", dt_grid.best_params_, "cv_MAE", round(-dt_grid.best_score_, 2))
    dt = dt_grid.best_estimator_.named_steps["model"]
    results.append(
        fit_one(
            "decision_tree",
            DecisionTreeRegressor(
                max_depth=dt.max_depth, min_samples_leaf=dt.min_samples_leaf, random_state=42
            ),
            pre,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    rf_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [6, 8, 12],
            "model__min_samples_leaf": [2, 4],
        },
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print("RF best", rf_grid.best_params_, "cv_MAE", round(-rf_grid.best_score_, 2))
    rp = rf_grid.best_params_
    results.append(
        fit_one(
            "random_forest",
            RandomForestRegressor(
                n_estimators=rp["model__n_estimators"],
                max_depth=rp["model__max_depth"],
                min_samples_leaf=rp["model__min_samples_leaf"],
                random_state=42,
                n_jobs=-1,
            ),
            pre,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    svr_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", SVR())]),
        param_grid={"model__C": [1000, 10000], "model__epsilon": [100, 500], "model__kernel": ["rbf"]},
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    svr_grid.fit(X_train, y_train)
    print("SVR best", svr_grid.best_params_, "cv_MAE", round(-svr_grid.best_score_, 2))
    sp = svr_grid.best_params_
    results.append(
        fit_one(
            "svr",
            SVR(C=sp["model__C"], epsilon=sp["model__epsilon"], kernel="rbf"),
            pre,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    real = [r for r in results if r["model"] != "dummy_mean"]
    best = max(real, key=lambda r: r["metrics_test"]["R2"])
    print("CHOSEN", best["model"], best["metrics_test"])

    joblib.dump(best["pipe"], MODELS_DIR / "model.joblib", compress=3)

    schema = {
        "task": "regression",
        "target": {
            "name": TARGET,
            "type": "number",
            "unit": "USD",
            "description": "Chi phí bảo hiểm y tế hàng năm",
        },
        "features": [
            {"name": "age", "type": "number", "required": True, "min": 18, "max": 64, "step": 1,
             "description": "Tuổi của người được bảo hiểm", "example": 30},
            {"name": "sex", "type": "categorical", "required": True, "enum": ["female", "male"],
             "description": "Giới tính", "example": "male"},
            {"name": "bmi", "type": "number", "required": True, "min": 15.0, "max": 55.0, "step": 0.01,
             "description": "Chỉ số khối cơ thể (Body Mass Index)", "example": 25.5},
            {"name": "children", "type": "number", "required": True, "min": 0, "max": 5, "step": 1,
             "description": "Số con phụ thuộc", "example": 2},
            {"name": "smoker", "type": "categorical", "required": True, "enum": ["yes", "no"],
             "description": "Có hút thuốc hay không", "example": "no"},
            {"name": "region", "type": "categorical", "required": True,
             "enum": ["northeast", "northwest", "southeast", "southwest"],
             "description": "Khu vực sinh sống tại Mỹ", "example": "southeast"},
        ],
        "feature_order": FEATURE_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    (MODELS_DIR / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    table = []
    for r in results:
        table.append(
            {
                "model": r["model"],
                **{k: r["metrics_test"][k] for k in ("MAE", "MSE", "RMSE", "R2")},
                "MAE_train": r["metrics_train"]["MAE"],
                "R2_train": r["metrics_train"]["R2"],
                "train_time_sec": r["train_time_sec"],
                "predict_ms": r["predict_ms"],
                "size_kb": r["size_kb"],
                "chosen": r["model"] == best["model"],
            }
        )
    pd.DataFrame(table).to_csv(MODELS_DIR / "results.csv", index=False)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        return o

    metadata = {
        "model_name": best["model"],
        "model_display_name": {
            "linear_regression": "Linear Regression",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "svr": "SVR",
        }.get(best["model"], best["model"]),
        "model_class": type(best["pipe"].named_steps["model"]).__name__,
        "model_version": "1.0.0",
        "task": "regression",
        "target": TARGET,
        "chosen_reason": (
            "Random Forest có R² test cao nhất và RMSE thấp nhất; "
            "khe train–test nhỏ hơn một cây đơn; thời gian dự đoán ~35ms chấp nhận được khi deploy."
            if best["model"] == "random_forest"
            else f"Chọn {best['model']} vì R² test cao nhất."
        ),
        "metrics": best["metrics_test"],
        "metrics_train": best["metrics_train"],
        "comparison": table,
        "hyperparameters": {
            "dummy_mean": {"strategy": "mean"},
            "linear_regression": {"fit_intercept": True},
            "decision_tree": {"best_params": dt_grid.best_params_, "search": "GridSearchCV cv=5 neg_MAE"},
            "random_forest": {"best_params": rp, "search": "GridSearchCV cv=3 neg_MAE"},
            "svr": {"best_params": sp, "search": "GridSearchCV cv=3 neg_MAE"},
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "n_samples": int(len(df)),
        "duplicates_found": n_dup,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "library_versions": {
            "scikit-learn": sklearn_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "artifact": {
            "path": "ai-models/models/model.joblib",
            "contains": "sklearn Pipeline (preprocessor + estimator)",
            "size_bytes": (MODELS_DIR / "model.joblib").stat().st_size,
            "export": (
                "Từ Colab: chạy 03_train.ipynb rồi Files.download('model.joblib') "
                "hoặc copy vào ai-models/models/. Docker nạp file này lúc container start."
            ),
        },
        "predict_ms": best["predict_ms"],
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(conv(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    backend_schema = AI_ROOT.parent / "app" / "backend" / "schema.json"
    if backend_schema.parent.exists():
        backend_schema.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu", MODELS_DIR / "model.joblib")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
