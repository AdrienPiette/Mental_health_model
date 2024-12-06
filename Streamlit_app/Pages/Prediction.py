import streamlit as st
import pickle
import pandas as pd

# Configure the Streamlit page
st.set_page_config(page_title="Mental Health Condition Prediction", page_icon="❤️", layout="wide")

# Introduction
st.title("Mental Health Condition Prediction")
st.markdown("""
###  
**Disclaimer**: This is for informational purposes only and should not replace medical advice from a healthcare professional.
""")

# Load the pre-trained model and training columns
model = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\Mental Health Model\Mental_health_model\Model_training\rf_model.pkl', 'rb'))
with open(r'C:\Users\pieta\OneDrive\Bureau\Mental Health Model\Mental_health_model\Model_training\encoder.pkl', 'rb') as f:
    train_columns = pickle.load(f)

# Collect user input features
st.write("Enter Your Details")
age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
job_role = st.selectbox("Job Role", ["Data Scientist", "Software Engineer", "Product Manager", "Business Analyst"])
industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Education"])
years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
work_location = st.selectbox("Work Location", ["Office", "Remote", "Hybrid"])
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
number_virtual_meetings = st.number_input("Number of Virtual Meetings", min_value=0, max_value=100, value=5)
work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=5, value=5)
stress_level = st.slider("Stress Level", min_value=1, max_value=5, value=3)
access_to_mental_healthcare = st.selectbox("Access to Mental Healthcare", ["Yes", "No"])
productivity_change = st.slider("Productivity Change", min_value=-5, max_value=5, value=0)
social_isolation_rating = st.slider("Social Isolation Rating", min_value=1, max_value=5, value=3)
satisfaction_with_remote_work = st.slider("Satisfaction with Remote Work", min_value=1, max_value=5, value=3)
company_support_for_remote_work = st.selectbox("Company Support for Remote Work", ["Yes", "No"])
physical_activity = st.slider("Physical Activity", min_value=1, max_value=5, value=3)
sleep_quality = st.slider("Sleep Quality", min_value=1, max_value=5, value=3)
region = st.selectbox("Region", ["North America", "Europe", "Asia", "Australia"])

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

# Align input data with training columns
input_dummies = input_dummies.reindex(columns=train_columns, fill_value=0)

# Display user input for review
st.write("### Your Input Data:")
st.write(input_dummies)

# Add a button to trigger prediction
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_dummies)

    # Display prediction result
    if prediction[0] == 1:
        st.markdown("## 🔴 Prediction: High Risk of Mental Health Condition 💔")
        st.markdown("It's recommended to consult a mental health professional for a detailed evaluation.")
    else:
        st.markdown("## 🟢 Prediction: Low Risk of Mental Health Condition 💪")
        st.markdown("Keep up the good habits and maintain a healthy lifestyle!")
