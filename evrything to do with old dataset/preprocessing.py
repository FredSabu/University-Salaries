import pandas as pd
import numpy as np

# Obviously change the file path
try:
    data = pd.read_csv('higher_ed_employee_salaries.csv')
except UnicodeDecodeError:
    data = pd.read_csv('higher_ed_employee_salaries.csv', encoding='ISO-8859-1')

# Handle Missing Values
for column in ['Job Description', 'Department']:
    if data[column].isnull().any():  # Check if there are any nulls in the column
        mode_value = data[column].mode()[0]
        data[column].fillna(mode_value, inplace=True)

# Normalize 'Earnings' if the distribution is skewed
if data['Earnings'].skew() > 1:
    data['Earnings'] = np.log1p(data['Earnings'])

# Save the preprocessed data to a new CSV file
data.to_csv('preprocessed_higher_ed_employee_salaries.csv', index=False)

# Display the first few rows of the processed data to verify changes
print(data.head())
print()
print("Data has been successfully preprocessed and saved.")
