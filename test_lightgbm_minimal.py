
import numpy as np
import lightgbm as lgb

print("LightGBM:", lgb.__version__)
print("NumPy:", np.__version__)

# Create a small test dataset
X = np.random.rand(100, 4).astype(np.float32)

y = np.array(
    [0, 1] * 50,
    dtype=np.int32
)

print("X shape:", X.shape)
print("X dtype:", X.dtype)
print("Y shape:", y.shape)
print("Y dtype:", y.dtype)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.05,
    verbosity=-1,
    n_jobs=1,
    device_type="cpu"
)

print("\nStarting LightGBM test...")

model.fit(X, y)

print("\nLightGBM TEST SUCCESSFUL!")

print("Predictions:", model.predict(X[:5]))
