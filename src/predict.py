from pathlib import Path
import joblib
import re
import pandas as pd
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

vectorizer = joblib.load(
    MODEL_DIR / "structured_tfidf.pkl"
)

encoder = joblib.load(
    MODEL_DIR / "structured_encoder.pkl"
)

model = joblib.load(
    MODEL_DIR / "calibrated_svm.pkl"
)

categorical_columns = joblib.load(
    MODEL_DIR / "categorical_columns.pkl"
)

binary_columns = joblib.load(
    MODEL_DIR / "binary_columns.pkl"
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_job(
    title,
    company_profile,
    description,
    requirements,
    benefits,
    location="Unknown",
    department="Unknown",
    employment_type="Unknown",
    required_experience="Unknown",
    required_education="Unknown",
    industry="Unknown",
    function="Unknown",
    telecommuting=0,
    has_company_logo=0,
    has_questions=0
):

    combined_text = (
        str(title) + " " +
        str(company_profile) + " " +
        str(description) + " " +
        str(requirements) + " " +
        str(benefits)
    )

    cleaned_text = clean_text(combined_text)

    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    structured_data = {
        "location": [location],
        "department": [department],
        "employment_type": [employment_type],
        "required_experience": [required_experience],
        "required_education": [required_education],
        "industry": [industry],
        "function": [function],
        "telecommuting": [telecommuting],
        "has_company_logo": [has_company_logo],
        "has_questions": [has_questions]
    }

    structured_df = pd.DataFrame(
        structured_data
    )

    categorical_data = encoder.transform(
        structured_df[categorical_columns]
    )

    binary_data = structured_df[
        binary_columns
    ].values

    final_features = hstack(
        [
            text_vector,
            categorical_data,
            binary_data
        ]
    ).tocsr()

    prediction = model.predict(
        final_features
    )[0]

    probabilities = model.predict_proba(
        final_features
    )[0]

    fraud_probability = probabilities[1]

    risk_score = fraud_probability * 100

    if prediction == 1:

        result = "Fraudulent"

        if risk_score >= 75:
            risk_level = "High Risk"
        elif risk_score >= 50:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

    else:

        result = "Genuine"

        if risk_score >= 50:
            risk_level = "Medium Risk"
        elif risk_score >= 25:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"

    return result, risk_score, risk_level


if __name__ == "__main__":

    result, risk_score, risk_level = predict_job(
        title="Software Developer",
        company_profile="ABC Technology Company",
        description="We are looking for a software developer with strong programming skills.",
        requirements="Knowledge of Python, Java, SQL and software development required.",
        benefits="Competitive salary and professional growth opportunities."
    )

    print("\nPrediction Result")
    print("----------------")
    print(f"Job Status: {result}")
    print(f"Risk Level: {risk_level}")
    print(f"Fraud Risk Score: {risk_score:.2f}%")