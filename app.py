import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model.pkl")

st.title("Loan Default Prediction")
st.write("Enter loan details to predict the loan status.")

loan_amount = st.number_input("Loan Amount", value=100000)
term = st.number_input("Term", value=360)
property_value = st.number_input("Property Value", value=150000)
income = st.number_input("Income", value=5000)
credit_score = st.number_input("Credit Score", value=700)
ltv = st.number_input("LTV", value=70.0)
dtir1 = st.number_input("DTIR1", value=40.0)

loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"])
approv_in_adv = st.selectbox("Approved in Advance", ["pre", "nopre"])
loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])
loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])
credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
open_credit = st.selectbox("Open Credit", ["opc", "nopc"])
business_or_commercial = st.selectbox(
    "Business or Commercial",
    ["b/c", "nob/c"]
)
neg_ammortization = st.selectbox(
    "Negative Amortization",
    ["neg_amm", "not_neg"]
)
interest_only = st.selectbox(
    "Interest Only",
    ["int_only", "not_int"]
)
lump_sum_payment = st.selectbox(
    "Lump Sum Payment",
    ["lpsm", "not_lpsm"]
)
occupancy_type = st.selectbox(
    "Occupancy Type",
    ["pr", "sr", "ir"]
)
total_units = st.selectbox(
    "Total Units",
    ["1U", "2U", "3U", "4U"]
)
co_applicant_credit_type = st.selectbox(
    "Co-applicant Credit Type",
    ["CIB", "EXP"]
)
age = st.selectbox(
    "Age",
    ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
)
submission = st.selectbox(
    "Submission of Application",
    ["to_inst", "not_inst"]
)
region = st.selectbox(
    "Region",
    ["North", "south", "central"]
)

if st.button("Predict"):
    data = pd.DataFrame([{
        "loan_amount": loan_amount,
        "term": term,
        "property_value": property_value,
        "income": income,
        "Credit_Score": credit_score,
        "LTV": ltv,
        "dtir1": dtir1,
        "loan_limit": loan_limit,
        "Gender": gender,
        "approv_in_adv": approv_in_adv,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "Credit_Worthiness": credit_worthiness,
        "open_credit": open_credit,
        "business_or_commercial": business_or_commercial,
        "Neg_ammortization": neg_ammortization,
        "interest_only": interest_only,
        "lump_sum_payment": lump_sum_payment,
        "occupancy_type": occupancy_type,
        "total_units": total_units,
        "co-applicant_credit_type": co_applicant_credit_type,
        "age": age,
        "submission_of_application": submission,
        "Region": region
    }])

    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("Prediction: Loan Default")
    else:
        st.success("Prediction: No Default")