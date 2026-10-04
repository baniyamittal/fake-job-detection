from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

X_train, X_test, y_train, y_test = joblib.load(
    MODEL_DIR / "structured_training_data.pkl"
)

models = {
    "structured_logistic_regression.pkl": LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        random_state=42
    ),
    "structured_naive_bayes.pkl": MultinomialNB(),
    "structured_random_forest.pkl": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "structured_svm.pkl": LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
}

for filename, model in models.items():
    print(f"\nTraining {filename}...")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / filename)
    print(f"Saved: {filename}")

svm = LinearSVC(
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

calibrated_svm = CalibratedClassifierCV(
    svm,
    cv=5,
    method="sigmoid"
)

print("\nTraining calibrated Linear SVM...")

calibrated_svm.fit(
    X_train,
    y_train
)

joblib.dump(
    calibrated_svm,
    MODEL_DIR / "calibrated_svm.pkl"
)

print("Saved: calibrated_svm.pkl")

print("\nModel Training Completed Successfully")