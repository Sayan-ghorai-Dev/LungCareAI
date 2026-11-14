
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

print("LightGBM:", lgb.__version__)

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")
df.columns = df.columns.str.strip().str.upper()
df = df.drop_duplicates().reset_index(drop=True)

# Target encoding
df["LUNG_CANCER"] = (
    df["LUNG_CANCER"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"NO": 0, "YES": 1})
)

df = df.dropna(subset=["LUNG_CANCER"])

X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"].astype(np.int32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocessing
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

X_train_processed = np.ascontiguousarray(
    preprocessor.fit_transform(X_train),
    dtype=np.float32
)

y_train_array = np.ascontiguousarray(
    y_train.to_numpy(),
    dtype=np.int32
)

weights = np.ascontiguousarray(
    compute_sample_weight("balanced", y_train_array),
    dtype=np.float32
)

print("\nData information:")
print("X:", X_train_processed.shape, X_train_processed.dtype)
print("y:", y_train_array.shape, y_train_array.dtype)
print("Weights:", weights.shape, weights.dtype)

def run_test(name, sample_weight=None):
    print("\nTesting:", name)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.05,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(
            X_train_processed,
            y_train_array,
            sample_weight=sample_weight
        )

        print("SUCCESS:", name)

    except Exception as error:
        print("FAILED:", name)
        print(type(error).__name__, error)

# Test 1: Actual data without weights
run_test("Actual data without sample weights")

# Test 2: Actual data with weights
run_test("Actual data with sample weights", weights)

# Test 3: Actual data with float32 labels
run_test(
    "Actual data with float32 labels",
    None
)

print("\nISOLATION TEST COMPLETED!")
