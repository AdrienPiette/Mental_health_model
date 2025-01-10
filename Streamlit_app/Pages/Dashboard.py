import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r'C:\Users\pieta\OneDrive\Bureau\Becode\bootcamp\Repositories\Mental Health Model\Mental_health_model\Data\data_clean_pre_encode.csv')

# Set up Streamlit page
st.set_page_config(page_title="Mental Health Dashboard", layout="wide")
st.title("Mental Health Condition Prediction Dashboard 🧠")

# Data Preview
st.subheader("Data Preview")
st.write(data.head())



# Define a color map for categorical values for 'Mental_Health_Condition'
color_map = {
    0: 'orange',  # No Mental Health Condition
    1: 'teal'     # Has Mental Health Condition
}

# Plot 1: Industry by Mental Health Condition
st.subheader("Industry by Mental Health Condition")
fig1 = px.histogram(
    data, 
    x='Industry', 
    color='Mental_Health_Condition', 
    barmode='group', 
    title='Distribution of Mental Health Condition by Industry',
    color_discrete_map=color_map
)
fig1.update_layout(xaxis_title='Industry', yaxis_title='Count', legend_title='Mental Health Condition')
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Industry by Company Support for Remote Work
st.subheader("Industry by Company Support for Remote Work")
fig2 = px.histogram(
    data, 
    x='Industry', 
    color='Company_Support_for_Remote_Work', 
    barmode='group', 
    title='Company Support for Remote Work by Industry',
    color_discrete_map=color_map
)
fig2.update_layout(xaxis_title='Industry', yaxis_title='Count', legend_title='Remote Work Support')
st.plotly_chart(fig2, use_container_width=True)

# Plot 3: Mental Health Condition by Region
st.subheader("Mental Health Condition by Region")
fig3 = px.histogram(
    data, 
    x='Region', 
    color='Mental_Health_Condition', 
    barmode='group', 
    title='Mental Health Condition by Region',
    color_discrete_map=color_map
)
fig3.update_layout(xaxis_title='Region', yaxis_title='Count', legend_title='Mental Health Condition')
st.plotly_chart(fig3, use_container_width=True)

# Boxplot 1: Mental Health Condition by Age and Gender
st.subheader("Mental Health Condition by Age and Gender")
fig4 = px.box(
    data, 
    x='Age', 
    y='Mental_Health_Condition', 
    color='Gender', 
    title='Mental Health Condition by Age and Gender',
    color_discrete_map={'Male': 'orange', 'Female': 'teal'}
)
fig4.update_layout(xaxis_title='Age', yaxis_title='Mental Health Condition Score', legend_title='Gender')
st.plotly_chart(fig4, use_container_width=True)

# Boxplot 2: Mental Health Condition by Age and Region
st.subheader("Mental Health Condition by Age and Region")
fig5 = px.box(
    data, 
    x='Age', 
    y='Mental_Health_Condition', 
    color='Region', 
    title='Mental Health Condition by Age and Region'
)
fig5.update_layout(xaxis_title='Age', yaxis_title='Mental Health Condition Score', legend_title='Region')
st.plotly_chart(fig5, use_container_width=True)

# Plot 4: Mental Health Condition by Stress Level and Work Location
st.subheader("Mental Health Condition by Stress Level and Work Location")
fig6 = px.histogram(
    data, 
    x='Stress_Level', 
    color='Work_Location', 
    barmode='group', 
    facet_col='Mental_Health_Condition', 
    title='Mental Health Condition by Stress Level and Work Location',
    color_discrete_map=color_map
)
fig6.update_layout(xaxis_title='Stress Level', yaxis_title='Count', legend_title='Work Location')
st.plotly_chart(fig6, use_container_width=True)

# Initialize the Label Encoder
encoder = LabelEncoder()

# Function to label-encode specified columns
columns_to_encode = ['Gender', 'Job_Role', 'Industry', 'Work_Location', 'Stress_Level', 'Mental_Health_Condition',
                     'Access_to_Mental_Health_Resources', 'Productivity_Change', 'Satisfaction_with_Remote_Work',
                     'Physical_Activity', 'Sleep_Quality', 'Region']

def label_encode(data, columns_to_encode):
    for col in columns_to_encode:
        data[col] = encoder.fit_transform(data[col])
    return data

# Apply label encoding to the dataset
data = label_encode(data, columns_to_encode)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = data.corr()
fig7 = px.imshow(
    corr_matrix.round(2), 
    text_auto=True, 
    aspect="auto", 
    color_continuous_scale='RdBu', 
    title='Correlation Heatmap'
)
fig7.update_layout(width=800, height=600)
st.plotly_chart(fig7, use_container_width=True)

# Most Correlated Features with Mental Health Condition
st.subheader("Most Correlated Features with Mental Health Condition")
corr_mhc = corr_matrix['Mental_Health_Condition'].sort_values(ascending=False)[1:]  # Exclude target variable
fig8 = px.bar(
    corr_mhc, 
    x=corr_mhc.index, 
    y=corr_mhc.values, 
    color=corr_mhc.values, 
    color_continuous_scale='Viridis', 
    title='Most Correlated Features with Mental Health Condition'
)
fig8.update_layout(
    xaxis_title='Feature',
    yaxis_title='Correlation',
    coloraxis_colorbar=dict(title='Correlation'),
    template='plotly_white',
    title_font_size=16,
    title_x=0.5
)
st.plotly_chart(fig8, use_container_width=True)

st.write("""Correlation Legend

This legend helps you understand the relationship between variables in the visualizations. The correlation coefficient indicates how strongly two variables are related:

    Positive Correlation (0 to +1): As one variable increases, the other tends to increase as well.
    Negative Correlation (0 to -1): As one variable increases, the other tends to decrease.
    No Correlation (around 0): There is no clear linear relationship between the two variables.

The closer the correlation coefficient is to 1 or -1, the stronger the relationship.""")