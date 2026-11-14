
import os
import traceback
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =====================================
# 1. CREATE FOLDERS
# =====================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# =====================================
# 2. LOAD DATASET
# =====================================

print("\nLoading LungCare AI Dataset...")

df = pd.read_csv("data/lung_cancer.csv")

df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_", regex=False)
)

df["GENDER"] = (
    df["GENDER"].astype(str).str.strip().str.upper()
)

df["LUNG_CANCER"] = (
    df["LUNG_CANCER"].astype(str).str.strip().str.upper()
)

# Remove duplicates
before = len(df)
df = df.drop_duplicates().copy()

print("Duplicates removed:", before - len(df))
print("Remaining records:", len(df))


# =====================================
# 3. FEATURES AND TARGET
# =====================================

X = df.drop(columns=["LUNG_CANCER"]).copy()

y = df["LUNG_CANCER"].map({
    "NO": 0,
    "YES": 1
})

if y.isnull().any():
    raise ValueError("Invalid target values found.")

# Convert target to NumPy
y = np.ascontiguousarray(
    y.to_numpy(dtype=np.float32).reshape(-1)
)

print("\nEncoded target values:", np.unique(y))

print("\nTarget Distribution:")
print(pd.Series(y).value_counts())


# =====================================
# 4. CHECK FEATURE DATA
# =====================================

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

X[numeric_columns] = X[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isnull().sum().sum() > 0:
    raise ValueError(
        "Missing or invalid feature values found."
    )

if not np.isfinite(
    X[numeric_columns].to_numpy(dtype=np.float64)
).all():
    raise ValueError("Infinite numeric feature values found.")


# =====================================
# 5. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Ensure contiguous arrays
y_train = np.ascontiguousarray(
    np.asarray(y_train, dtype=np.float32).reshape(-1)
)

y_test = np.ascontiguousarray(
    np.asarray(y_test, dtype=np.float32).reshape(-1)
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# =====================================
# 6. SAMPLE WEIGHTS
# =====================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

sample_weights = np.ascontiguousarray(
    np.asarray(sample_weights, dtype=np.float32).reshape(-1)
)

print("\nSample weights generated successfully!")


# =====================================
# 7. PREPROCESSOR FUNCTION
# =====================================

def create_preprocessor():

    return ColumnTransformer(
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


# =====================================
# 8. DEFINE MODELS
# =====================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        class_weight=None,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
        device_type="cpu",
        force_col_wise=True
    )
}


# =====================================
# 9. TRAIN AND EVALUATE
# =====================================

results = []
trained_models = {}

for name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Training", name)
    print("=" * 50)

    try:

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", classifier)
        ])

        # Train model
        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights
        )

        print(name, "training completed!")

        # Predictions
        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Evaluation
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro_F1": macro_f1,
            "ROC_AUC": auc
        })

        trained_models[name] = pipeline

        print("\nEvaluation Results:")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("Macro F1:", round(macro_f1, 4))
        print("ROC-AUC:", round(auc, 4))

    except Exception as error:

        print("\nERROR while training", name)
        print("Error:", str(error))

        traceback.print_exc()

        print("\nSkipping", name, "and continuing...")


# =====================================
# 10. CHECK TRAINED MODELS
# =====================================

if len(trained_models) == 0:

    raise RuntimeError(
        "No model was trained successfully."
    )

print("\nSuccessfully trained models:")
print(list(trained_models.keys()))


# =====================================
# 11. MODEL COMPARISON
# =====================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)


# =====================================
# 12. SELECT BEST MODEL
# =====================================

best_name = results_df.loc[
    results_df["Macro_F1"].idxmax(),
    "Model"
]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)


# =====================================
# 13. SAVE TRAINED MODELS
# =====================================

model_files = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl"
}

for name, pipeline in trained_models.items():

    path = os.path.join(
        "models",
        model_files[name]
    )

    joblib.dump(
        pipeline,
        path
    )

    print(name, "saved successfully:", path)


# Save best model separately
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully!")


# =====================================
# 14. CLASSIFICATION REPORT
# =====================================

best_predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["NO", "YES"],
    zero_division=0
)

with open(
    "reports/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("Best Model: " + best_name + "\n\n")
    file.write(report)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)


# =====================================
# 15. CONFUSION MATRIX
# =====================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=[0, 1],
    display_labels=["NO", "YES"],
    cmap="Blues"
)

plt.title("LungCare AI - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)

plt.close()


# =====================================
# 16. MODEL COMPARISON CHART
# =====================================

plot_df = results_df.set_index("Model")[
    ["Accuracy", "Macro_F1", "ROC_AUC"]
]

plot_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("LungCare AI Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "reports/model_comparison.png",
    dpi=300
)

plt.close()


# =====================================
# 17. FINAL STATUS
# =====================================

print("\n" + "=" * 50)
print("LUNGCARE AI TRAINING COMPLETED!")
print("=" * 50)

print("Successfully Trained Models:", len(trained_models))
print("Best Model:", best_name)

print("\nModels saved in: models/")
print("Reports saved in: reports/")

print("\nGenerated Files:")
print("- models/random_forest.pkl (if trained)")
print("- models/xgboost.pkl (if trained)")
print("- models/lightgbm.pkl (if trained)")
print("- models/best_model.pkl")
print("- reports/model_comparison.csv")
print("- reports/classification_report.txt")
print("- reports/confusion_matrix.png")
print("- reports/model_comparison.png")

print("\nEducational ML prototype only.")
print("Not a medical diagnostic tool.")

print("\nTRAINING FINISHED SUCCESSFULLY!")
