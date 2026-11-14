
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

print("Loading actual LungCare AI dataset...")

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

y = np.ascontiguousarray(
    y.to_numpy(dtype=np.int32)
)

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

print("\nApplying preprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)

X_train_processed = np.ascontiguousarray(
    X_train_processed,
    dtype=np.float32
)

y_train = np.ascontiguousarray(
    y_train,
    dtype=np.int32
)

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

weights = np.ascontiguousarray(
    weights,
    dtype=np.float32
)

print("Processed X:", X_train_processed.shape)
print("X dtype:", X_train_processed.dtype)
print("Y dtype:", y_train.dtype)
print("Weights dtype:", weights.dtype)

print("\nTraining actual dataset with LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
    n_jobs=1,
    device_type="cpu",
    force_col_wise=True
)

model.fit(
    X_train_processed,
    y_train,
    sample_weight=weights
)

print("\nREAL DATASET LIGHTGBM TEST SUCCESSFUL!")

print("Accuracy:", model.score(
    preprocessor.transform(X_test),
    y_test
))
