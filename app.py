import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="wide")

# Title and description
st.title("🏦 Bank Customer Churn Prediction System")
st.markdown("---")

# Sidebar for model training status
with st.sidebar:
    st.header("About")
    st.info("This system predicts whether a bank customer is likely to churn (leave the bank) based on their profile information.")
    st.markdown("**Model:** XGBoost Classifier")
    st.markdown("**Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Has Credit Card, Active Member Status, Estimated Salary")

# Load and train model
@st.cache_resource
def train_model():
    try:
        # Load dataset
        dataset = pd.read_csv("Churn_Modelling.csv")
        dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        
        # Encode categorical variables
        encoder_geo = LabelEncoder()
        encoder_gender = LabelEncoder()
        dataset["Geography"] = encoder_geo.fit_transform(dataset["Geography"])
        dataset["Gender"] = encoder_gender.fit_transform(dataset["Gender"])
        
        # Prepare features and target
        X = dataset.drop("Exited", axis=1)
        y = dataset["Exited"]
        
        # Scale features
        scaler = MinMaxScaler()
        bumpy_features = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
        df_scaled = pd.DataFrame(data=X)
        df_scaled[bumpy_features] = scaler.fit_transform(X[bumpy_features])
        X = df_scaled
        
        # Apply SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=7)
        
        # Train model
        clf = XGBClassifier(max_depth=12, random_state=7, n_estimators=100, 
                           min_child_weight=3, colsample_bytree=0.75, subsample=0.8)
        clf.fit(X_train, y_train)
        
        return clf, scaler, encoder_geo, encoder_gender, True
    except FileNotFoundError:
        return None, None, None, None, False
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None, None, False

# Load model
with st.spinner("Loading model... Please wait"):
    model, scaler, encoder_geo, encoder_gender, model_loaded = train_model()

if not model_loaded:
    st.error("⚠️ Error: 'Customer-Churn-Records.csv' file not found in the current directory!")
    st.info("Please make sure the CSV file is in the same folder as this script.")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Input form
st.header("📝 Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1,
                                   help="Customer's credit score (300-850)")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                            help="Customer's country")
    gender = st.selectbox("Gender", ["Male", "Female"],
                         help="Customer's gender")

with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1,
                         help="Customer's age")
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1,
                            help="Number of years with the bank")
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0,
                             help="Current account balance")

with col3:
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"],
                              help="Does customer have a credit card?")
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"],
                                   help="Is customer actively using services?")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0,
                                      help="Customer's estimated annual salary")

st.markdown("---")

# Predict button
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Prepare input data
    geography_encoded = encoder_geo.transform([geography])[0]
    gender_encoded = encoder_gender.transform([gender])[0]
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val = 1 if is_active_member == "Yes" else 0
    
    # Create input dataframe with all features
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography_encoded],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [1],  # Default value
        'HasCrCard': [has_cr_card_val],
        'IsActiveMember': [is_active_val],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Scale the features
    bumpy_features = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    input_data_scaled = input_data.copy()
    input_data_scaled[bumpy_features] = scaler.transform(input_data[bumpy_features])
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK: Customer is likely to CHURN")
            st.markdown(f"**Churn Probability:** {prediction_proba[1]*100:.2f}%")
        else:
            st.success("### ✅ LOW RISK: Customer is likely to STAY")
            st.markdown(f"**Stay Probability:** {prediction_proba[0]*100:.2f}%")
    
    with result_col2:
        st.metric("Confidence Score", f"{max(prediction_proba)*100:.2f}%")
    
    # Customer details summary
    with st.expander("📋 Customer Details Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Credit Score:** {credit_score}")
            st.write(f"**Geography:** {geography}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Age:** {age}")
            st.write(f"**Tenure:** {tenure} years")
        with summary_col2:
            st.write(f"**Balance:** ${balance:,.2f}")
            st.write(f"**Has Credit Card:** {has_cr_card}")
            st.write(f"**Active Member:** {is_active_member}")
            st.write(f"**Estimated Salary:** ${estimated_salary:,.2f}")
    
    # Display probability distribution
    st.markdown("---")
    st.subheader("📈 Probability Distribution")
    
    prob_df = pd.DataFrame({
        'Outcome': ['Stay', 'Churn'],
        'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
    })
    st.bar_chart(prob_df.set_index('Outcome'))
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    if prediction == 1:
        st.warning("""
        **⚠️ High Churn Risk - Immediate Action Required:**
        - 📞 **Reach out immediately** with retention offers
        - 🔍 **Investigate** reasons for potential dissatisfaction
        - 🎁 **Offer personalized benefits** or incentives
        - 👤 **Schedule a follow-up call** with relationship manager
        - 💳 **Review account activity** for warning signs
        - 🎯 **Priority retention** campaign enrollment
        """)
    else:
        st.info("""
        **✅ Low Churn Risk - Maintain Engagement:**
        - ⭐ **Continue providing** excellent service
        - 📊 **Monitor account activity** regularly
        - 🚀 **Consider upselling** additional products
        - 📧 **Maintain engagement** through regular communication
        - 🎖️ **Reward loyalty** with exclusive offers
        - 📈 **Track satisfaction** metrics
        """)

# Footer
st.markdown("---")
st.markdown("*Developed with Streamlit & XGBoost | Bank Churn Prediction System v1.0*")