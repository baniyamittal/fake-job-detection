# 🛡️ Fake Job Posting Detection System

A machine learning-based web application that detects potentially fraudulent job postings using textual information and structured job attributes.

The system analyzes job postings using Natural Language Processing (NLP), TF-IDF feature extraction, structured feature encoding, and multiple machine learning algorithms. It provides a fraud prediction, calibrated fraud-risk score, risk level, prediction history, and model analytics through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Fake and fraudulent job postings can mislead job seekers and may result in financial loss, identity theft, or exposure to malicious activities.

The objective of this project is to develop an intelligent system that can analyze job postings and identify patterns associated with fraudulent listings.

The system combines:

- Textual job information
- Categorical job attributes
- Binary job attributes
- TF-IDF feature extraction
- Machine learning classification
- Calibrated fraud-risk estimation
- User authentication
- Prediction history
- Model performance analytics

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze job postings and identify fraudulent patterns.
2. Clean and preprocess textual job information.
3. Extract meaningful textual features using TF-IDF.
4. Incorporate structured job attributes into the prediction system.
5. Train and compare multiple machine learning models.
6. Evaluate models using multiple performance metrics.
7. Develop a functional system for predicting unseen job postings.
8. Provide an interpretable fraud-risk score.
9. Store prediction history for registered users.
10. Provide a professional dashboard for analysis and monitoring.

---

## ✨ Key Features

### 🔐 User Authentication

- User registration
- User login
- Password validation
- Session-based authentication
- Logout functionality

### 🔍 Fake Job Detection

Users can enter:

- Job title
- Location
- Department
- Employment type
- Required experience
- Required education
- Industry
- Job function
- Company profile
- Job description
- Requirements
- Benefits
- Telecommuting status
- Company logo availability
- Screening questions availability

The system then predicts whether the job is:

- ✅ Genuine
- 🚨 Fraudulent

---

## 📊 Fraud Risk Score

The application provides a calibrated fraud-risk estimate.

Risk categories:

| Risk Score | Risk Level |
|---|---|
| 0% – 24.99% | Very Low Risk |
| 25% – 49.99% | Low Risk |
| 50% – 74.99% | Medium Risk |
| 75% – 100% | High Risk |

The risk score represents the model's estimated probability of fraudulent-job risk and should not be treated as absolute proof.

---

## 🧠 Machine Learning Methodology

The project follows the following pipeline:

```text
Raw Job Dataset
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Text Preprocessing
       ↓
TF-IDF Feature Extraction
       ↓
Structured Feature Encoding
       ↓
Feature Combination
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Calibrated Prediction
       ↓
Fraud Risk Score