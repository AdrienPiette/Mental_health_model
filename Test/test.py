import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\pieta\OneDrive\Bureau\Mental Health Model\Mental_health_model\Data\Impact_of_Remote_Work_on_Mental_Health_cleaned.csv')

# Specify the column name
column_name = 'Job_Role'

# Print distinct values of the specified column
distinct_values = df[column_name].unique()
print(f"Distinct values in column '{column_name}':")
print(distinct_values)