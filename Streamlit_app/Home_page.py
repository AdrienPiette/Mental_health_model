import streamlit as st
from flask import Flask
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello from Flask! Visit /run_streamlit to start Streamlit.'

@app.route('/run_streamlit')
def run_streamlit():
    # Run the Streamlit app as a subprocess
    subprocess.Popen(['streamlit', 'run', 'Home_page.py'])
    return "Streamlit is running on a separate port!"

if __name__ == "__main__":
    # Run Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)
    
def run_streamlit_app():

# Set page configuration
   st.set_page_config(
      page_title="Mental Health Condition Prediction Dashboard",
      page_icon="🧠",
      layout="wide"
   )

   # Add the title and introductory text
   st.title("Mental Health Condition Prediction Dashboard 🧠")

   st.markdown("""
   Welcome to the **Mental Health Condition Prediction Dashboard**! 🌈  
   This platform aims to support mental well-being by assessing the likelihood of mental health conditions based on individual factors and behavioral patterns.  
   Utilizing advanced **machine learning models** and validated datasets, this tool provides valuable insights for both individuals and healthcare professionals. 💡

   ---

   ### **Features of the Dashboard:**
   1. **Real-Time Predictions**:
      - Input personal and behavioral data to receive a risk assessment. 📋
   2. **Data Visualization**:
      - Discover trends in mental health statistics and contributing factors. 📊
   3. **Interactivity**:
      - Customize input data and explore predictions dynamically. ⚙️

   ### **How It Works:**
   1. Input data such as age, lifestyle habits, stress levels, and sleep patterns.
   2. Our trained machine learning model analyzes the data and predicts mental health risk.
   3. Leverage visualizations to gain insights into factors influencing mental health.

   Together, let’s break the stigma around mental health and promote wellness! 🌟  
   """)

   # Add an image related to mental health
   st.image(
      "https://upload.wikimedia.org/wikipedia/commons/3/3d/Brain_%28lightbulb%29.png",
      caption="Predict Mental Health Conditions with Machine Learning",
      width=600
   )

   # Call to action
   st.markdown("""
   ---
   ### **Get Started**
   Use the sidebar to:
   - **Make Predictions**: Enter data to assess mental health risks. 📝
   - **View Insights**: Explore data visualizations and trends. 📈
   - **Learn More**: Understand how the prediction model works and how it can help. 💡

   **Your mental health matters. Let’s take steps toward understanding and improving it together!** 💜
   """)

if __name__ == "__main__":
    # Run the Streamlit part of the app when this file is run directly
    run_streamlit_app()
