
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("Loading dataset...")

df = pd.read_csv("data/lung_cancer.csv")

df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_", regex=False)
)

df["GENDER"] = df["GENDER"].astype(str).str.strip().str.upper()
df["LUNG_CANCER"] = df["LUNG_CANCER"].astype(str).str.strip().str.upper()

df = df.drop_duplicates().copy()

X = df.drop(columns=["LUNG_CANCER"])

y = df["LUNG_CANCER"].map({
    "NO": 0,
    "YES": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "gender",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32
            ),
            ["GENDER"]
        )
    ],
    remainder="passthrough"
)

X_actual = np.ascontiguousarray(
    preprocessor.fit_transform(X_train),
    dtype=np.float32
)

y_actual = np.ascontiguousarray(
    y_train.to_numpy(dtype=np.float32)
)

X_synthetic = np.random.default_rng(42).random(
    X_actual.shape,
    dtype=np.float32
)

y_synthetic = np.array(
    [0, 1] * (len(y_actual) // 2),
    dtype=np.float32
)

if len(y_synthetic) < len(y_actual):
    y_synthetic = np.append(y_synthetic, np.float32(0))

mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic_x"

if mode == "synthetic_x":
    X_test_data = X_synthetic
    y_test_data = y_actual
elif mode == "synthetic_y":
    X_test_data = X_actual
    y_test_data = np.ascontiguousarray(y_synthetic)
else:
    raise ValueError("Use synthetic_x or synthetic_y")

print("\nTest mode:", mode)
print("X shape:", X_test_data.shape)
print("X dtype:", X_test_data.dtype)
print("Y shape:", y_test_data.shape)
print("Y dtype:", y_test_data.dtype)
print("Y distribution:", np.unique(y_test_data, return_counts=True))

model = lgb.LGBMClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.05,
    verbosity=-1,
    n_jobs=1,
    device_type="cpu",
    force_col_wise=True
)

print("\nStarting diagnostic training...")

model.fit(X_test_data, y_test_data)

print("\nDIAGNOSTIC TEST PASSED!")
