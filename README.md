# CreditWise Loan System

## Overview

CreditWise Loan System is an end-to-end Machine Learning application designed to assist financial institutions in making faster, more accurate, and unbiased loan approval decisions.

The project addresses the challenges faced by traditional manual loan verification processes, where loan officers evaluate applications based on income, employment details, credit history, and supporting documents. Manual assessment can lead to inconsistencies, delays, and incorrect decisions that impact both customers and financial institutions.

This system leverages historical loan application data and Machine Learning algorithms to predict whether a loan application should be approved or rejected before final human verification.

---
Check it out here-
[Launch Application](https://creditwise-loan-system-rg.streamlit.app/)

## Problem Statement

A financial institution receives hundreds of loan applications daily from customers across urban and rural regions. The traditional manual approval process often results in:

* Rejection of creditworthy applicants.
* Approval of high-risk applicants.
* Delays in processing applications.
* Human bias and inconsistent decision-making.

The objective of this project is to develop an intelligent loan approval system capable of learning patterns from historical data and generating reliable approval predictions.

---

## Features

* End-to-end Machine Learning pipeline
* Interactive web application using Streamlit
* Automated loan approval prediction
* Data preprocessing and feature engineering
* Handling categorical and numerical features
* Model evaluation and comparison
* Real-time predictions through a user-friendly interface

---

## Dataset Features

The dataset contains applicant information including:

* Applicant Income
* Coapplicant Income
* Employment Status
* Age
* Marital Status
* Dependents
* Credit Score
* Existing Loans
* Debt-to-Income Ratio (DTI)
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Loan Purpose
* Property Area
* Education Level
* Gender
* Employer Category

Target Variable:

* Loan Approved (Yes / No)

---

## Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Label Encoding
* One-Hot Encoding
* Feature Scaling using StandardScaler

### Feature Engineering

Additional features were created to improve model performance:

* DTI_Ratio_sq
* Credit_Score_sq

### Models Implemented

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Gaussian Naive Bayes

### Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report
* Model Comparison

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

## Project Structure

```text
CreditWise Loan System/
│
├── app.py
├── loan_model.pkl
├── ohe.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── CreditWiseLoanSystem.ipynb
└── loan_approval_data.csv
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd CreditWise-Loan-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

---

## Results

The system was trained and evaluated using multiple classification algorithms. Model performance was compared, and the best-performing model was deployed through a Streamlit web application for real-time loan approval prediction.

---

## Author

Roney Ghosh


