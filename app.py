import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MAZA AI", layout="wide")
st.title("3-in-1 Maze Financial Intelligence")
st.markdown("---")

maze_choice = st.sidebar.selectbox(
    "Choose a Project to Evaluate", 
    ["Credit Risk Assessment", "Financial Stress Scoring", "Savings Goal Predictor"]
)
     
model_map = {
    "Credit Risk Assessment": "models/credit_risk_pipeline.pkl",
    "Financial Stress Scoring": "models/financial_stress_pipeline.pkl",
    "Savings Goal Predictor": "models/savings_goal_pipeline.pkl"
}

def load_active_model(path):
    return joblib.load(path)

try:
    model = load_active_model(model_map[maze_choice])
    
    st.header(f"Predictor: {maze_choice}")
    st.info("The model will automatically process your inputs through the saved pipeline.")

    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        monthly_expense_total = st.number_input("Total Monthly Expenses ($)", min_value=0, value=3000)
        savings_rate = st.slider("Savings Rate (%)", 0, 100, 20) / 100
        investment_amount = st.number_input("Current Investment Amount ($)", min_value=0, value=1000)

    with col2:
        transaction_count = st.number_input("Number of Monthly Transactions", min_value=0, value=30)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
        if maze_choice == "Credit Risk Assessment":
            employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
            loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Home", "Medical"])
        
        elif maze_choice == "Financial Stress Scoring":
            housing_type = st.selectbox("Housing Type", ["Own", "Rent", "Mortgage"])
            num_dependents = st.number_input("Number of Dependents", min_value=0, value=0)

        elif maze_choice == "Savings Goal Predictor":
            goal_amount = st.number_input("Target Savings Goal ($)", min_value=0, value=5000)
            time_horizon = st.number_input("Time Horizon (Months)", min_value=1, value=12)

    if st.button("Generate AI Prediction"):
        data_dict = {
            'monthly_income': [monthly_income],
            'monthly_expense_total': [monthly_expense_total],
            'savings_rate': [savings_rate],
            'investment_amount': [investment_amount],
            'transaction_count': [transaction_count],
            'age': [age]
        }
        
        if maze_choice == "Credit Risk Assessment":
            data_dict['employment_status'] = [employment_status]
            data_dict['loan_purpose'] = [loan_purpose]
        elif maze_choice == "Financial Stress Scoring":
             data_dict['housing_type'] = [housing_type]
             data_dict['num_dependents'] = [num_dependents]
        elif maze_choice == "Savings Goal Predictor":
             data_dict['goal_amount'] = [goal_amount]
             data_dict['time_horizon'] = [time_horizon]
        
        input_df = pd.DataFrame(data_dict)
        
        prediction = model.predict(input_df)
    
        st.markdown("---")
        st.subheader("Results")
        
        if maze_choice == "Credit Risk Assessment":
            risk_labels = {0: "Poor", 1: "Average", 2: "Good"}
            result = risk_labels.get(prediction[0], f"Class {prediction[0]}")
            st.metric(label="Credit Risk Tier", value=result)
        
        elif maze_choice == "Financial Stress Scoring":
            st.metric(label="Predicted Stress Level", value=f"Level {prediction[0]}")
            
        else: 
            status = "Likely to Meet Goal" if prediction[0] == 1 else "Unlikely to Meet Goal"
            st.metric(label="Savings Outcome", value=status)

except FileNotFoundError:
    st.error(f"Error: Could not find the model file at {model_map[maze_choice]}. Please ensure the 'models' folder is on GitHub.")
