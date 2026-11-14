
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split

print("LightGBM:", lgb.__version__)

# Load actual target
df = pd.read_csv("data/lung_cancer.csv")
df.columns = df.columns.str.strip().str.upper()
df = df.drop_duplicates().reset_index(drop=True)

y = (
    df["LUNG_CANCER"].astype(str).str.strip().str.upper()
    .map({"NO": 0, "YES": 1})
)

y = y.dropna().astype(np.int32)

_, _, _, y_train = train_test_split(
    np.zeros((len(y), 1)),
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

actual_y = np.array(y_train, dtype=np.int32, copy=True)

# Create labels with the same class counts
balanced_order_y = np.array(
    [0] * 30 + [1] * 190,
    dtype=np.int32
)

# Shuffle the same labels
shuffled_y = balanced_order_y.copy()
np.random.default_rng(42).shuffle(shuffled_y)

rng = np.random.default_rng(42)

X = np.ascontiguousarray(
    rng.random((220, 16)).astype(np.float32)
)

def test(name, labels):
    print("\nTesting:", name)
    print("Counts:", np.unique(labels, return_counts=True))
    print("Dtype:", labels.dtype)
    print("Shape:", labels.shape)
    print("Contiguous:", labels.flags["C_CONTIGUOUS"])

    try:
        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(X.copy(), labels.copy())

        print("SUCCESS:", name)

    except Exception as e:
        print("FAILED:", name)
        print(type(e).__name__, e)

test("Generated labels: 30 zeros, 190 ones", balanced_order_y)

test("Shuffled generated labels", shuffled_y)

test("Actual dataset labels", actual_y)

print("\nTARGET TEST COMPLETED!")
