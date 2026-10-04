from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    df[column] = df[column].fillna("")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

df["combined_text"] = df["combined_text"].apply(clean_text)

output_path = BASE_DIR / "data" / "processed_jobs.csv"
df.to_csv(output_path, index=False)

print("\nPreprocessing Completed")
print(f"Original Shape: {df.shape}")
print(f"Processed File: {output_path}")
print("\nSample Processed Text:")
print(df["combined_text"].iloc[0][:1000])