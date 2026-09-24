"""
Pipeline tiền xử lý dùng chung lúc train và lúc phục vụ.
Fit CHỈ trên tập train — tránh rò rỉ dữ liệu.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "charges"

SRC_DIR = Path(__file__).resolve().parent
AI_ROOT = SRC_DIR.parent
DATA_ZIP = AI_ROOT / "data" / "dataset.zip"
DATA_CSV = AI_ROOT / "data" / "insurance.csv"


def load_raw_dataframe() -> pd.DataFrame:
    """Đọc dataset.zip (bắt buộc) — fallback sang csv nếu đang làm việc local."""
    if DATA_ZIP.exists():
        with ZipFile(DATA_ZIP) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as fh:
                df = pd.read_csv(fh)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    else:
        raise FileNotFoundError(f"Không tìm thấy {DATA_ZIP} hoặc {DATA_CSV}")
    return df


def inspect_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra ban đầu + làm sạch, có lý do:

    - Missing: dataset Kaggle này không thiếu giá trị. Vẫn gắn SimpleImputer
      trong pipeline để service không gãy nếu một request thiếu số.
    - Duplicate: có 1 dòng trùng. Giữ nguyên để khớp 1338 bản ghi công bố
      trên Kaggle; 1/1338 không đổi metric đáng kể.
    - Giá trị vô lý: age 18–64, children 0–5, bmi ~16–53, charges > 0.
      Không cắt ngoại lai vì chi phí cao của người hút thuốc là tín hiệu thật
      (xem EDA hình 2, 4), không phải nhiễu đo lường.
    """
    expected = FEATURE_ORDER + [TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột: {missing_cols}")
    return df[expected].copy()


def split_xy(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Chia TRƯỚC khi fit scaler/encoder.
    Hồi quy nên không stratify. random_state cố định để lặp lại được.
    Tỉ lệ 80/20: tập test ~268 dòng, đủ ổn định cho MAE/RMSE, còn 80% để GridSearchCV.
    """
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric  -> median imputer + StandardScaler
               (cần cho SVR / Linear; cây không bắt buộc nhưng để CÙNG pipeline
                cho mọi model, so sánh công bằng).
    Categorical -> most_frequent imputer + OneHotEncoder
               (sex/smoker/region là danh mục không có thứ tự → One-Hot,
                không dùng Ordinal).
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
