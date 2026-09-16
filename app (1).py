import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna()
    return df


@st.cache_resource
def train_model():

    df = load_data()

    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train_processed, y_train)

    return model, preprocessor


model, preprocessor = train_model()

st.divider()

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.subheader("💳 Account Information")

col3, col4 = st.columns(2)

with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col4:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

if st.button("🔮 Predict Churn", use_container_width=True):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    customer_processed = preprocessor.transform(customer)

    prediction = model.predict(customer_processed)[0]

    probability = model.predict_proba(
        customer_processed
    )[0][1]

    st.divider()

    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ HIGH CHURN RISK — Customer is likely to churn."
        )

        st.write(
            "Recommended action: Consider retention offers, "
            "customer support intervention, or plan review."
        )

    else:

        st.success(
            "✅ LOW CHURN RISK — Customer is unlikely to churn."
        )

        st.write(
            "The customer currently shows a lower probability "
            "of leaving the company."
        )

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

st.divider()

st.caption(
    "Customer Churn Prediction using Logistic Regression | "
    "Business Analytics Project"
)