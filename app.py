import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ------------------ Load Model ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ Mapping Dictionary ------------------
mapping = {
    "gender": {"Female": 1, "Male": 2},
    "Partner": {"Yes": 1, "No": 2},
    "Dependents": {"No": 1, "Yes": 2},
    "PhoneService": {"No": 1, "Yes": 2},
    "MultipleLines": {"No phone service": 1, "No": 2, "Yes": 3},
    "InternetService": {"DSL": 1, "No": 2, "Fiber optic": 3},
    "OnlineSecurity": {"No internet service": 1, "No": 2, "Yes": 3},
    "OnlineBackup": {"No internet service": 1, "No": 2, "Yes": 3},
    "DeviceProtection": {"No internet service": 1, "No": 2, "Yes": 3},
    "TechSupport": {"No internet service": 1, "No": 2, "Yes": 3},
    "StreamingTV": {"No internet service": 1, "No": 2, "Yes": 3},
    "StreamingMovies": {"No internet service": 1, "No": 2, "Yes": 3},
    "Contract": {"Month-to-month": 1, "One year": 2, "Two year": 3},
    "PaperlessBilling": {"Yes": 1, "No": 2},
    "PaymentMethod": {
        "Electronic check": 1,
        "Mailed check": 2,
        "Bank transfer (automatic)": 3,
        "Credit card (automatic)": 4,
    }
}

def convert(col, val):
    return mapping[col][val]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>📞 Telco Customer Churn Prediction Dashboard</h1>
    <p style='text-align:center; font-size:18px;'>A smart ML-powered tool to predict customer churn and explore your dataset visually.</p>
    <hr style="border:1px solid #4CAF50;">
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
menu = st.sidebar.radio("📌 Navigation", ["🔮 Predict Churn", "📊 Dataset Explorer", "📈 Insights Dashboard"])


# -----------------------------------------------------------------
# 🔮 PREDICTION PAGE
# -----------------------------------------------------------------
if menu == "🔮 Predict Churn":
    st.subheader("🔮 Customer Churn Prediction Form")
    st.write("Fill the details below to check if a customer is likely to churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])

    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "No", "Fiber optic"])
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    col4, col5, col6 = st.columns(3)
    with col4:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col5:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col6:
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    if st.button("✨ Predict"):
        data = np.array([[
            convert("gender", gender),
            convert("Partner", Partner),
            convert("Dependents", Dependents),
            convert("PhoneService", PhoneService),
            convert("MultipleLines", MultipleLines),
            convert("InternetService", InternetService),
            convert("OnlineSecurity", OnlineSecurity),
            convert("OnlineBackup", OnlineBackup),
            convert("DeviceProtection", DeviceProtection),
            convert("TechSupport", TechSupport),
            convert("StreamingTV", StreamingTV),
            convert("StreamingMovies", StreamingMovies),
            convert("Contract", Contract),
            convert("PaperlessBilling", PaperlessBilling),
            convert("PaymentMethod", PaymentMethod),
            tenure,
            MonthlyCharges,
            TotalCharges,
            0  # dummy
        ]])

        pred = model.predict(data)[0]

        if pred == 1:
            st.error("🚨 **Customer is Likely to Churn**")
        else:
            st.success("💚 **Customer will NOT Churn**")



# -----------------------------------------------------------------
# 📊 DATASET EXPLORER
# -----------------------------------------------------------------
elif menu == "📊 Dataset Explorer":
    st.subheader("📊 Dataset Viewer")
    uploaded = st.file_uploader("Upload your Telco dataset (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### 🔍 Data Preview")
        st.dataframe(df)

        st.write("### 📌 Summary Statistics")
        st.dataframe(df.describe())



# -----------------------------------------------------------------
# 📈 INSIGHTS DASHBOARD
# -----------------------------------------------------------------
elif menu == "📈 Insights Dashboard":
    st.subheader("📈 Data Insights & Visualizations")

    uploaded = st.file_uploader("Upload dataset to view insights", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="MonthlyCharges", title="Monthly Charges Distribution")
            st.plotly_chart(fig)

        with col2:
            fig = px.pie(df, names="gender", title="Gender Breakdown")
            st.plotly_chart(fig)

        st.write("### 🔥 Churn by Contract Type")
        fig = px.bar(df, x="Contract", color="Churn")
        st.plotly_chart(fig)
