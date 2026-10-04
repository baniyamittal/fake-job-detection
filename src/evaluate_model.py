from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

X_train, X_test, y_train, y_test = joblib.load(
    MODEL_DIR / "structured_training_data.pkl"
)

models = {
    "Logistic Regression": joblib.load(
        MODEL_DIR / "structured_logistic_regression.pkl"
    ),
    "Naive Bayes": joblib.load(
        MODEL_DIR / "structured_naive_bayes.pkl"
    ),
    "Random Forest": joblib.load(
        MODEL_DIR / "structured_random_forest.pkl"
    ),
    "Linear SVM": joblib.load(
        MODEL_DIR / "structured_svm.pkl"
    )
}

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Genuine",
                "Fraudulent"
            ],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Genuine",
            "Fraudulent"
        ]
    )

    display.plot()

    plt.title(
        f"{name} - Structured Features"
    )

    figure_path = REPORT_DIR / (
        "structured_"
        + name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("STRUCTURED MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    REPORT_DIR / "structured_model_comparison.csv",
    index=False
)

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Accuracy: {best_model['Accuracy']:.4f}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall: {best_model['Recall']:.4f}"
)

print(
    f"F1 Score: {best_model['F1 Score']:.4f}"
)

print(
    f"ROC-AUC: {best_model['ROC-AUC']:.4f}"
)

print("\nEvaluation completed successfully.")