import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load("loan_model.pkl")
ohe = joblib.load("ohe.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Predictor")

# Numerical Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18)
Dependents = st.number_input("Dependents", min_value=0)
Credit_Score = st.number_input("Credit Score", min_value=0.0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)
Savings = st.number_input("Savings", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term", min_value=1)

# Categorical Inputs
Employment_Status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed", "Contract"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Business", "Education", "Personal", "Car"]
)

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

Education_Level = st.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Government", "Private", "Business", "MNC", "Unemployed"]
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Applicant_Income": [Applicant_Income],
        "Coapplicant_Income": [Coapplicant_Income],
        "Employment_Status": [Employment_Status],
        "Age": [Age],
        "Marital_Status": [Marital_Status],
        "Dependents": [Dependents],
        "Credit_Score": [Credit_Score],
        "Existing_Loans": [Existing_Loans],
        "DTI_Ratio": [DTI_Ratio],
        "Savings": [Savings],
        "Collateral_Value": [Collateral_Value],
        "Loan_Amount": [Loan_Amount],
        "Loan_Term": [Loan_Term],
        "Loan_Purpose": [Loan_Purpose],
        "Property_Area": [Property_Area],
        "Education_Level": [Education_Level],
        "Gender": [Gender],
        "Employer_Category": [Employer_Category]
    })

    # Label Encoding
    input_df["Education_Level"] = input_df["Education_Level"].map({
        "Graduate": 0,
        "Not Graduate": 1
    })

    # One Hot Encoding
    cat_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    encoded = ohe.transform(input_df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols)
    )

    input_df = pd.concat(
        [input_df.drop(columns=cat_cols), encoded_df],
        axis=1
    )

    # Feature Engineering
    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
    input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")