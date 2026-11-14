
import numpy as np
import lightgbm as lgb

print("LightGBM:", lgb.__version__)
print("NumPy:", np.__version__)

rng = np.random.default_rng(42)

# Method 1: float64 -> float32
X1 = rng.random((220, 16)).astype(np.float32)
y1 = np.array([0, 1] * 110, dtype=np.int32)

# Method 2: Direct float32 generation
X2 = rng.random((220, 16), dtype=np.float32)
y2 = np.array([0, 1] * 110, dtype=np.int32)

# Method 3: float64 -> float32 + actual labels
df = __import__("pandas").read_csv("data/lung_cancer.csv")
df.columns = df.columns.str.strip().str.upper()
df = df.drop_duplicates().reset_index(drop=True)

y_actual = (
    df["LUNG_CANCER"].astype(str).str.strip().str.upper()
    .map({"NO": 0, "YES": 1})
    .dropna()
    .astype(np.int32)
)

from sklearn.model_selection import train_test_split

_, _, _, y_train = train_test_split(
    np.zeros((len(y_actual), 1)),
    y_actual,
    test_size=0.20,
    random_state=42,
    stratify=y_actual
)

y3 = y_train.to_numpy(dtype=np.int32)

def test(name, X, y):
    print("\nTesting:", name)
    print("X dtype:", X.dtype, "shape:", X.shape)
    print("y dtype:", y.dtype, "shape:", y.shape)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(X, y)
        print("SUCCESS:", name)

    except Exception as e:
        print("FAILED:", name)
        print(type(e).__name__, e)

test("Float64 converted to float32", X1, y1)
test("Direct float32 generation", X2, y2)
test("Float64 converted + actual labels", X1, y3)

print("\nDTYPE TEST COMPLETED!")
