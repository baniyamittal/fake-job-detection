from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed_jobs.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

text_column = "combined_text"

categorical_columns = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown").astype(str)

for column in binary_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)

X_text = df[text_column].fillna("")

X_structured = df[
    categorical_columns + binary_columns
]

y = df["fraudulent"]

X_train_text, X_test_text, y_train, y_test, X_train_structured, X_test_structured = train_test_split(
    X_text,
    y,
    X_structured,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_categorical = encoder.fit_transform(
    X_train_structured[categorical_columns]
)

X_test_categorical = encoder.transform(
    X_test_structured[categorical_columns]
)

X_train_binary = X_train_structured[
    binary_columns
].values

X_test_binary = X_test_structured[
    binary_columns
].values

X_train_final = hstack(
    [
        X_train_tfidf,
        X_train_categorical,
        X_train_binary
    ]
).tocsr()

X_test_final = hstack(
    [
        X_test_tfidf,
        X_test_categorical,
        X_test_binary
    ]
).tocsr()

joblib.dump(
    tfidf,
    MODEL_DIR / "structured_tfidf.pkl"
)

joblib.dump(
    encoder,
    MODEL_DIR / "structured_encoder.pkl"
)

joblib.dump(
    (
        X_train_final,
        X_test_final,
        y_train,
        y_test
    ),
    MODEL_DIR / "structured_training_data.pkl"
)

joblib.dump(
    categorical_columns,
    MODEL_DIR / "categorical_columns.pkl"
)

joblib.dump(
    binary_columns,
    MODEL_DIR / "binary_columns.pkl"
)

print("\nStructured Feature Engineering Completed")

print(f"Training Samples: {X_train_final.shape[0]}")
print(f"Testing Samples: {X_test_final.shape[0]}")
print(f"Total Features: {X_train_final.shape[1]}")

print(
    f"Training Fraudulent Jobs: {y_train.sum()}"
)

print(
    f"Testing Fraudulent Jobs: {y_test.sum()}"
)

print("\nText Features:", X_train_tfidf.shape[1])

print(
    "Categorical Features:",
    X_train_categorical.shape[1]
)

print(
    "Binary Features:",
    len(binary_columns)
)

print("\nFeature engineering completed successfully.")