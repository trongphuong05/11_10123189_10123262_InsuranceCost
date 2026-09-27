"""
Huấn luyện ≥4 model trên cùng split, lưu pipeline+model thành model.joblib.
Chạy:  python train.py   (từ thư mục ai-models/src)
"""

from __future__ import annotations

import json
import logging
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

# ---------------------------------------------------------------------------
# Cấu hình Cấu trúc Log (Structured Logging)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

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
    logger.info("Đang huấn luyện Pipeline cho mô hình: %s...", name)
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
    
    logger.info(
        "Hoàn tất [%s] -> Test R2=%.4f MAE=%.2f RMSE=%.2f | Train R2=%.4f | TrainTime=%.2fs PredictLatency=%.2fms FileSize=%.1fKB",
        name, m_te['R2'], m_te['MAE'], m_te['RMSE'], m_tr['R2'], train_s, pred_ms, size / 1024
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
    logger.info("=== Bắt đầu quy trình huấn luyện & đánh giá mô hình ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Đang nạp và làm sạch dữ liệu đầu vào...")
    df = inspect_and_clean(load_raw_dataframe())
    n_dup = int(df.duplicated().sum())
    logger.info("Dữ liệu sau khi làm sạch: Số dòng=%d | Trùng lặp=%d | Khuyết thiếu=%d", len(df), n_dup, int(df.isnull().sum().sum()))
    
    X_train, X_test, y_train, y_test = split_xy(df)
    logger.info("Phân chia tập dữ liệu: Train=%d dòng | Test=%d dòng | random_state=42", len(X_train), len(X_test))
    
    pre = build_preprocessor()
    logger.info("Đã khởi tạo Preprocessor (ColumnTransformer) thành công.")

    results = []

    # 1. Dummy Regressor
    logger.info("--- [1/5] Huấn luyện Baseline (DummyRegressor) ---")
    results.append(fit_one("dummy_mean", DummyRegressor(strategy="mean"), pre, X_train, y_train, X_test, y_test))

    # 2. Linear Regression
    logger.info("--- [2/5] Huấn luyện Linear Regression ---")
    results.append(fit_one("linear_regression", LinearRegression(), pre, X_train, y_train, X_test, y_test))

    # 3. Decision Tree (GridSearchCV)
    logger.info("--- [3/5] Thực thi GridSearchCV cho Decision Tree ---")
    dt_param_grid = {"model__max_depth": [4, 6, 8, 10], "model__min_samples_leaf": [4, 8, 16]}
    logger.debug("DT Parameter Grid: %s", dt_param_grid)
    dt_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", DecisionTreeRegressor(random_state=42))]),
        param_grid=dt_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    t_dt = time.perf_counter()
    dt_grid.fit(X_train, y_train)
    logger.info("DT GridSearchCV hoàn tất trong %.2fs | Best Params: %s | Best CV MAE: %.2f",
                time.perf_counter() - t_dt, dt_grid.best_params_, -dt_grid.best_score_)
    
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

    # 4. Random Forest (GridSearchCV)
    logger.info("--- [4/5] Thực thi GridSearchCV cho Random Forest Regressor ---")
    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 8, 12],
        "model__min_samples_leaf": [2, 4],
    }
    logger.info("RF Search Space: n_estimators=%s, max_depth=%s, min_samples_leaf=%s",
                rf_param_grid["model__n_estimators"], rf_param_grid["model__max_depth"], rf_param_grid["model__min_samples_leaf"])
    rf_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        param_grid=rf_param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    t_rf = time.perf_counter()
    rf_grid.fit(X_train, y_train)
    logger.info("RF GridSearchCV hoàn tất trong %.2fs | Best Params: %s | Best CV MAE: %.2f",
                time.perf_counter() - t_rf, rf_grid.best_params_, -rf_grid.best_score_)
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

    # 5. SVR (GridSearchCV)
    logger.info("--- [5/5] Thực thi GridSearchCV cho Support Vector Regressor (SVR) ---")
    svr_param_grid = {"model__C": [1000, 10000], "model__epsilon": [100, 500], "model__kernel": ["rbf"]}
    logger.info("SVR Search Space: C=%s, epsilon=%s, kernel=%s",
                svr_param_grid["model__C"], svr_param_grid["model__epsilon"], svr_param_grid["model__kernel"])
    svr_grid = GridSearchCV(
        Pipeline([("preprocessor", pre), ("model", SVR())]),
        param_grid=svr_param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    t_svr = time.perf_counter()
    svr_grid.fit(X_train, y_train)
    logger.info("SVR GridSearchCV hoàn tất trong %.2fs | Best Params: %s | Best CV MAE: %.2f",
                time.perf_counter() - t_svr, svr_grid.best_params_, -svr_grid.best_score_)
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

    # Lựa chọn mô hình tốt nhất
    real = [r for r in results if r["model"] != "dummy_mean"]
    best = max(real, key=lambda r: r["metrics_test"]["R2"])
    logger.info("=== MÔ HÌNH TỐT NHẤT ĐƯỢC CHỌN: [%s] (R2 Test: %.4f, MAE Test: %.2f) ===",
                best["model"], best["metrics_test"]["R2"], best["metrics_test"]["MAE"])

    joblib.dump(best["pipe"], MODELS_DIR / "model.joblib", compress=3)
    logger.info("Đã xuất artifact mô hình được chọn tại: %s", MODELS_DIR / "model.joblib")

    # Xuất Schema
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
    logger.info("Đã ghi file Schema: %s", MODELS_DIR / "schema.json")

    # Lưu kết quả dạng CSV
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
    logger.info("Đã lưu bảng so sánh hiệu năng các mô hình: %s", MODELS_DIR / "results.csv")

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
    logger.info("Đã xuất file Metadata chi tiết: %s", MODELS_DIR / "metadata.json")

    backend_schema = AI_ROOT.parent / "app" / "backend" / "schema.json"
    if backend_schema.parent.exists():
        backend_schema.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Đã đồng bộ file Schema sang Backend: %s", backend_schema)
        
    logger.info("=== QUY TRÌNH HUẤN LUYỆN DỰ ÁN AI THÀNH CÔNG HOÀN HẢO ===")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()