
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("LightGBM:", lgb.__version__)

# Load dataset
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

# Split both X and y together
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
)

y_actual = np.ascontiguousarray(
    y_train.to_numpy(),
    dtype=np.int32
)

rng = np.random.default_rng(42)

X_random = np.ascontiguousarray(
    rng.random((len(y_actual), X_actual.shape[1])).astype(np.float32)
)

y_balanced = np.array(
    [0, 1] * (len(y_actual) // 2),
    dtype=np.int32
)

def test(name, X_data, y_data):
    print("\nTesting:", name)
    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(X_data, y_data)
        print("SUCCESS:", name)

    except Exception as e:
        print("FAILED:", name)
        print(type(e).__name__, e)

test("Random X + actual y", X_random, y_actual)

test("Actual X + balanced y", X_actual, y_balanced)

test("Actual X + actual y", X_actual, y_actual)

print("\nFINAL TEST COMPLETED!")

