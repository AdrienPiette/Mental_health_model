import streamlit as st

# Configurer la page
st.set_page_config(
    page_title="Mental Health Condition Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Ajouter le titre et le texte introductif
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



# Appel à l'action
st.markdown("""
---
### **Get Started**
Use the sidebar to:
- **Make Predictions**: Enter data to assess mental health risks. 📝
- **View Insights**: Explore data visualizations and trends. 📈
- **Learn More**: Understand how the prediction model works and how it can help. 💡

**Your mental health matters. Let’s take steps toward understanding and improving it together!** 💜
""")
