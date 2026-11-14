
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("LightGBM:", lgb.__version__)

# Load actual dataset
df = pd.read_csv("data/lung_cancer.csv")
df.columns = df.columns.str.strip().str.upper()
df = df.drop_duplicates().reset_index(drop=True)

df["LUNG_CANCER"] = (
    df["LUNG_CANCER"].astype(str).str.strip().str.upper()
    .map({"NO": 0, "YES": 1})
)

df = df.dropna(subset=["LUNG_CANCER"])

X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"].astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("gender", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32
        ), ["GENDER"])
    ],
    remainder="passthrough"
)

X_actual = np.ascontiguousarray(
    preprocessor.fit_transform(X_train),
    dtype=np.float32
).copy()

y_actual = np.array(
    y_train.to_numpy(),
    dtype=np.int32,
    order="C",
    copy=True
)

rng = np.random.default_rng(42)

X_random = np.ascontiguousarray(
    rng.random((220, 16), dtype=np.float32)
)

y_balanced = np.array(
    [0, 1] * 110,
    dtype=np.int32
)

y_float64 = y_actual.astype(np.float64).copy()

print("\nActual labels:")
print("Counts:", np.unique(y_actual, return_counts=True))
print("Shape:", y_actual.shape)
print("Dtype:", y_actual.dtype)
print("Contiguous:", y_actual.flags["C_CONTIGUOUS"])
print("Own data:", y_actual.flags["OWNDATA"])
print("Unique values:", np.unique(y_actual))

def run_test(name, X_data, y_data):
    print("\nTesting:", name)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(
            np.ascontiguousarray(X_data).copy(),
            np.ascontiguousarray(y_data).copy()
        )

        print("SUCCESS:", name)

    except Exception as error:
        print("FAILED:", name)
        print(type(error).__name__, error)

run_test("Random features + actual labels", X_random, y_actual)

run_test("Random features + balanced labels", X_random, y_balanced)

run_test("Actual features + balanced labels", X_actual, y_balanced)

run_test("Random features + actual float64 labels", X_random, y_float64)

print("\nLABEL TEST COMPLETED!")
