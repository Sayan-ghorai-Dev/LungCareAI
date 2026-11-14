
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("Loading actual dataset...")

df = pd.read_csv("data/lung_cancer.csv")
df.columns = df.columns.str.strip().str.upper()
df = df.drop_duplicates().reset_index(drop=True)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
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

X_processed = np.ascontiguousarray(
    preprocessor.fit_transform(X_train),
    dtype=np.float32
)

y_array = np.ascontiguousarray(
    y_train.to_numpy(),
    dtype=np.int32
)

feature_names = preprocessor.get_feature_names_out()

print("\nShape:", X_processed.shape)
print("Data type:", X_processed.dtype)
print("Labels:", np.unique(y_array, return_counts=True))

print("\nChecking feature values...")

print("NaN values:", np.isnan(X_processed).sum())
print("Positive infinity:", np.isposinf(X_processed).sum())
print("Negative infinity:", np.isneginf(X_processed).sum())
print("All finite:", np.isfinite(X_processed).all())

for i, name in enumerate(feature_names):
    col = X_processed[:, i]

    print(
        f"{i}: {name} | "
        f"min={np.nanmin(col)} | "
        f"max={np.nanmax(col)} | "
        f"unique={len(np.unique(col))} | "
        f"finite={np.isfinite(col).all()}"
    )

def test_model(name, data):
    print("\nTesting:", name)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(data, y_array)
        print("SUCCESS:", name)

    except Exception as error:
        print("FAILED:", name)
        print(type(error).__name__, error)

# Test 1: Original actual features
test_model("Original actual features", X_processed)

# Test 2: Replace invalid values
X_clean = np.nan_to_num(
    X_processed,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_clean = np.ascontiguousarray(X_clean, dtype=np.float32)

test_model("Cleaned actual features", X_clean)

# Test 3: Test each feature individually
print("\nTesting individual columns...")

for i, name in enumerate(feature_names):
    print("\nFeature:", name)

    test_model(
        str(name),
        np.ascontiguousarray(X_clean[:, i:i+1])
    )

print("\nFEATURE TEST COMPLETED!")
