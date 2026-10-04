# 🔍 Fake Job Posting Detection System

An end-to-end Machine Learning web application that detects whether a job posting is **Genuine or Fraudulent** using textual and structured job information.

The system combines **TF-IDF text features** with structured job-posting features and uses multiple Machine Learning algorithms to identify potentially fraudulent job postings.

---

## 🚀 Live Demo

🌐 **Try the deployed application:**

https://fake-job-detection-ktu3cj79gaw5cnpwzjw8cv.streamlit.app/

The application provides:

- User Registration
- Secure Login
- Job Fraud Detection
- Fraud Risk Score
- Prediction History
- Model Analytics
- User Profile

---

## 📂 GitHub Repository

💻 **Source Code:**

https://github.com/baniyamittal/fake-job-detection

---

## 🎯 Project Objective

Online job platforms can contain fraudulent job advertisements designed to collect personal information, money, or sensitive data from job seekers.

The objective of this project is to develop a Machine Learning system that can:

- Analyze job postings
- Identify suspicious job advertisements
- Classify jobs as Genuine or Fraudulent
- Calculate a fraud risk score
- Provide a user-friendly prediction interface
- Maintain prediction history
- Compare different Machine Learning models

---

## ✨ Key Features

### 🔐 User Authentication

- User registration
- User login
- Password validation
- User profile
- Logout functionality

### 🕵️ Fake Job Detection

Users can enter:

- Job title
- Company profile
- Job description
- Requirements
- Benefits
- Location
- Department
- Employment type
- Required experience
- Required education
- Industry
- Function
- Telecommuting
- Company logo availability
- Questions availability

The system predicts:

**Genuine** or **Fraudulent**

---

### ⚠️ Fraud Risk Score

The system generates a fraud probability and converts it into a risk score.

Risk levels include:

- Very Low Risk
- Low Risk
- Medium Risk
- High Risk

This allows users to understand the severity of the prediction instead of seeing only a binary result.

---

### 📊 Prediction History

Every prediction made by a logged-in user can be stored with:

- Job title
- Prediction result
- Risk level
- Fraud risk score
- Date and time

---

### 📈 Model Analytics

The application provides model performance information using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

The system compares:

- Logistic Regression
- Naive Bayes
- Random Forest
- Linear SVM

---

## 🧠 Machine Learning Methodology

The project follows the following pipeline:

```text
Raw Job Posting
       ↓
Data Preprocessing
       ↓
Text Cleaning
       ↓
Feature Engineering
       ↓
TF-IDF Text Features
       +
Structured Features
       ↓
Train/Test Split
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Fraud Prediction
       ↓
Risk Score
