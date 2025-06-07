from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0055aa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    </style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_resource
def load_model(path="model/model_v2.joblib"):
    """Load and cache the trained model."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model_v2.joblib' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_transformers(path="transformers"):
    """Load the scalers and encoders"""
    try:
        transformers={
            "contract_ohe":joblib.load(path + "/contract_ohe.pkl"),
            "int_service_ohe":joblib.load(path + "/int_service_ohe.pkl"),
            "monthly_charges_scaler":joblib.load(path + "/monthly_charges_scaler.pkl"),
            "payment_method_ohe":joblib.load(path + "/payment_method_ohe.pkl"),
            "tenure_scaler":joblib.load(path + "/tenure_scaler.pkl"),
            "total_charges_scaler":joblib.load(path + "/total_charges_scaler.pkl")
        }
        return transformers
    except Exception as e:
        st.error(f"Error loading transformer: {str(e)}")
        return None

def preprocess_input(input_dict):
    """Convert raw inputs into the numeric DataFrame the model expects."""
    df = pd.DataFrame([input_dict])
    transformers = load_transformers()
    print(transformers)
    # Map binary and categorical features
    mapping = {
        "Yes": 1, "No": 0,
        "No phone service": 0, "No internet service": 0,
        "Male": 1, "Female": 0
    }

    df["InternetService"] = df["InternetService"].replace({"No":"No Service"})
    df = df.replace(mapping)

    # Scaling
    df["MonthlyCharges"] = transformers["monthly_charges_scaler"].transform(df[["MonthlyCharges"]])
    df["TotalCharges"] = transformers["total_charges_scaler"].transform(df[["TotalCharges"]])
    df["tenure"] = transformers["tenure_scaler"].transform(df[["tenure"]])

    # One hot encoding
    internet_service_ohe = pd.DataFrame(transformers["int_service_ohe"].transform(df[["InternetService"]]), columns=transformers["int_service_ohe"].get_feature_names_out())
    contract_ohe = pd.DataFrame(transformers["contract_ohe"].transform(df[["Contract"]]), columns=transformers["contract_ohe"].get_feature_names_out())
    payment_method_ohe = pd.DataFrame(transformers["payment_method_ohe"].transform(df[["PaymentMethod"]]), columns=transformers["payment_method_ohe"].get_feature_names_out())
    df = pd.concat([df, internet_service_ohe, contract_ohe, payment_method_ohe], axis=1)
    df = df.drop(columns=["InternetService", "PaymentMethod", "Contract"])

    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.infer_objects(copy=False)
    return df

# Input validation
def validate_inputs(input_dict):
    """Validate input data before prediction."""
    required_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for field in required_fields:
        if input_dict[field] is None or np.isnan(input_dict[field]):
            return False, f"Please provide a valid {field} value"
    return True, ""

# Sidebar
with st.sidebar:
    st.header("👤 Customer Profile")

    with st.expander("Personal Information", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior = st.radio("Senior Citizen?", ["No", "Yes"], key="senior")
        partner = st.radio("Has Partner?", ["No", "Yes"], key="partner")
        dependents = st.radio("Has Dependents?", ["No", "Yes"], key="dependents")

    st.markdown("---")
    with st.expander("Usage & Services", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
        phone = st.radio("Phone Service", ["Yes", "No"], key="phone")
        mlines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="mlines")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="online_sec")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="online_backup")
        device_prof = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device_prof")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech_support")
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stream_tv")
        stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="stream_movies")

    st.markdown("---")
    with st.expander("Contract & Billing", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless = st.radio("Paperless Billing", ["Yes", "No"], key="paperless")
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment"
        )

    st.markdown("---")
    with st.expander("Charges", expanded=True):
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.05, key="monthly")
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0, step=0.1, key="total")

# Main content
st.title("📊 Customer Churn Prediction")
st.write("Complete the customer profile in the sidebar and click to Predict churn.")

# Prediction
if st.button("Predict Churn"):
    with st.spinner("Analyzing customer profile..."):
        # Gather inputs
        print("Contract", contract)
        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": mlines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prof,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        # Validate inputs
        is_valid, error_message = validate_inputs(raw_input)
        if not is_valid:
            st.error(error_message)
        else:
            model = load_model()
            if model:
                try:
                    X = preprocess_input(raw_input)
                    print([i for i in X], [X.iloc[0]])
                    proba = model.predict(X.to_numpy())
                    # proba = 0.7
                    churn = "Yes" if proba >= 0.5 else "No"

                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h4>Churn?</h4>
                                <h2>{churn}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h4>Churn Probability</h4>
                                <h2>{float(proba):.1%}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(proba * 100))

                    # Advice box
                    st.markdown("---")
                    if churn == "Yes":
                        st.error("⚠️ This customer is likely to churn. Consider retention offers or reach out proactively.")
                        with st.expander("Retention Suggestions"):
                            st.write("- Offer a discount on monthly charges")
                            st.write("- Provide premium features for free")
                            st.write("- Schedule a customer satisfaction call")
                    else:
                        st.success("✅ This customer is likely to stay. Keep up the good service!")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    raise e

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | By Rhutvik Hegde")
