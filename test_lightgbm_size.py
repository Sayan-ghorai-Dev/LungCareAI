
import numpy as np
import lightgbm as lgb

print("LightGBM:", lgb.__version__)

# Generate identical synthetic data
rng = np.random.default_rng(42)

X = rng.random((220, 16)).astype(np.float32)

y = np.array(
    [0, 1] * 110,
    dtype=np.int32
)

# Test different dataset sizes
sizes = [50, 100, 150, 200, 220]

for size in sizes:

    print("\n" + "=" * 40)
    print("Testing dataset size:", size)

    X_test = np.ascontiguousarray(X[:size])
    y_test = np.ascontiguousarray(y[:size])

    try:

        model = lgb.LGBMClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.05,
            verbosity=-1,
            n_jobs=1,
            device_type="cpu"
        )

        model.fit(X_test, y_test)

        print("SUCCESS:", size, "records")

    except Exception as error:

        print("FAILED:", size, "records")
        print("Error:", error)
        break

print("\nSIZE TEST COMPLETED!")
