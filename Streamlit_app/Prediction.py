import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure the Streamlit page
st.set_page_config(page_title="Mental Health Condition Prediction", page_icon="❤️", layout="wide")

# Introduction
st.title("Mental Health Condition Prediction")
st.markdown("""
###  
**Disclaimer**: This is for informational purposes only and should not replace medical advice from a healthcare professional.
""")

# Load the pre-trained model, scaler, and training columns
try:
    with open(r'C:\Users\pieta\OneDrive\Bureau\Becode\bootcamp\Repositories\Mental Health Model\Mental_health_model\Model_training\rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(r'C:\Users\pieta\OneDrive\Bureau\Becode\bootcamp\Repositories\Mental Health Model\Mental_health_model\Model_training\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(r'C:\Users\pieta\OneDrive\Bureau\Becode\bootcamp\Repositories\Mental Health Model\Mental_health_model\Model_training\training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
except Exception as e:
    st.error("Failed to load model or preprocessing files. Please check the file paths.")
    st.stop()

# Collect user input features
st.write("### Enter Your Details")

# Age input
age = st.number_input("Age", min_value=1, max_value=100, value=25, help="Enter your age in years.")

# Gender input
gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender.")

# Job Role input
job_role = st.selectbox("Job Role", ['HR', 'Data Scientist', 'Software Engineer', 'Sales', 'Marketing', 'Designer','Project Manager'], help="Select your current job role.")



# Industry input
industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Education"], help="Select the industry you work in.")

# Years of Experience input
years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1, help="Enter the number of years you have been working in your field.")

# Work Location input
work_location = st.selectbox("Work Location", ["Office", "Remote", "Hybrid"], help="Select your current work location.")

# Hours per Week input
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, help="Enter the average number of hours you work per week.")

# Number of Virtual Meetings input
number_virtual_meetings = st.number_input("Number of Virtual Meetings", min_value=0, max_value=100, value=5, help="Enter the average number of virtual meetings you have per week.")

# Work-Life Balance input
work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=5, value=5, help="Rate your work-life balance on a scale of 1 to 5.")

# Stress Level input
stress_level = st.slider("Stress Level", min_value=1, max_value=5, value=3, help="Rate your current stress level on a scale of 1 to 5.")

# Access to Mental Healthcare input
access_to_mental_healthcare = st.selectbox("Access to Mental Healthcare", ["Yes", "No"], help="Do you have access to mental healthcare resources?")

# Productivity Change input
productivity_change = st.slider("Productivity Change", min_value=-5, max_value=5, value=0, help="Rate the change in your productivity on a scale of -5 to 5.")

# Social Isolation Rating input
social_isolation_rating = st.slider("Social Isolation Rating", min_value=1, max_value=5, value=3, help="Rate your level of social isolation on a scale of 1 to 5.")

# Satisfaction with Remote Work input
satisfaction_with_remote_work = st.slider("Satisfaction with Remote Work", min_value=1, max_value=5, value=3, help="Rate your satisfaction with remote work on a scale of 1 to 5.")

# Company Support for Remote Work input
company_support_for_remote_work = st.selectbox("Company Support for Remote Work", ["Yes", "No"], help="Does your company support remote work?")

# Physical Activity input
physical_activity = st.slider("Physical Activity", min_value=1, max_value=5, value=3, help="Rate your level of physical activity on a scale of 1 to 5.")

# Sleep Quality input
sleep_quality = st.slider("Sleep Quality", min_value=1, max_value=5, value=3, help="Rate your sleep quality on a scale of 1 to 5.")

# Region input
region = st.selectbox("Region", ["North America", "Europe", "Asia", "Australia"], help="Select the region you are currently living in.")

# Prepare input data
input_data = {
    'Age': age,
    'Gender': gender,
    'Job_Role': job_role,
    'Industry': industry,
    'Years_Experience': years_experience,
    'Work_Location': work_location,
    'Hours_Worked_Per_Week': hours_per_week,
    'Number_of_Virtual_Meetings': number_virtual_meetings,
    'Work_Life_Balance_Rating': work_life_balance,
    'Stress_Level': stress_level,
    'Access_to_Mental_Health_Resources': access_to_mental_healthcare,
    'Productivity_Change': productivity_change,
    'Social_Isolation_Rating': social_isolation_rating,
    'Satisfaction_with_Remote_Work': satisfaction_with_remote_work,
    'Company_Support_for_Remote_Work': company_support_for_remote_work,
    'Physical_Activity': physical_activity,
    'Sleep_Quality': sleep_quality,
    'Region': region
}

# Convert input into a pandas DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Encode categorical columns with pd.get_dummies and align with training columns
input_dummies = pd.get_dummies(input_df)
input_dummies = input_dummies.reindex(columns=training_columns, fill_value=0)

# Scale input data
input_scaled = scaler.transform(input_dummies)



# Add a button to trigger prediction
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_scaled)

        # Display prediction result
        if prediction[0] == 1:
            st.markdown("## 🔴 Prediction: High Risk of Mental Health Condition 💔")
            st.markdown("It's recommended to consult a mental health professional for a detailed evaluation.")
        else:
            st.markdown("## 🟢 Prediction: Low Risk of Mental Health Condition 💪")
            st.markdown("Keep up the good habits and maintain a healthy lifestyle!")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

