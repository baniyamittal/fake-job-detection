from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed_jobs.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df["combined_text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

joblib.dump(
    (X_train_tfidf, X_test_tfidf, y_train, y_test),
    MODEL_DIR / "training_data.pkl"
)

print("\nTF-IDF Feature Engineering Completed")
print(f"Training Samples: {X_train_tfidf.shape[0]}")
print(f"Testing Samples: {X_test_tfidf.shape[0]}")
print(f"TF-IDF Features: {X_train_tfidf.shape[1]}")
print(f"Training Fraudulent Jobs: {y_train.sum()}")
print(f"Testing Fraudulent Jobs: {y_test.sum()}")