import pandas as pd
import numpy as np

# Define a function to categorize job descriptions (example logic)
def categorize_job(title):
    if 'professor' in title.lower():
        return 'Professor'
    # Add more categorization rules as needed
    return title

# Load the data, handling encoding issues
try:
    data = pd.read_csv('/Users/Mathi/Desktop/university_salaries.csv')
except UnicodeDecodeError:
    data = pd.read_csv('/Users/Mathi/Desktop/university_salaries.csv', encoding='ISO-8859-1')

# Fill missing 'Job Description' and 'Department' with mode
for column in ['Job Description', 'Department']:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Impute missing 'Earnings' with median
data['Earnings'].fillna(data['Earnings'].median(), inplace=True)

# Normalize 'Earnings' with log transformation to reduce skewness
if data['Earnings'].skew() > 1:
    data['Earnings'] = np.log1p(data['Earnings'])

# Categorize 'Job Description' for better visualization
data['Job Description'] = data['Job Description'].apply(categorize_job)

# Check for consistency in 'School' and 'Department'
# Normalize the case and strip extra whitespace
data['School'] = data['School'].str.lower().str.strip()
data['Department'] = data['Department'].str.lower().str.strip()

# Detect and handle outliers in 'Earnings'
# Calculate IQR
Q1 = data['Earnings'].quantile(0.25)
Q3 = data['Earnings'].quantile(0.75)
IQR = Q3 - Q1

# Define thresholds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Handle outliers
data['Earnings'] = np.where(data['Earnings'] > upper_bound, upper_bound, data['Earnings'])
data['Earnings'] = np.where(data['Earnings'] < lower_bound, lower_bound, data['Earnings'])

# Save the preprocessed data to a new CSV file
data.to_csv('/Users/Mathi/Desktop/preprocessed_university_salaries.csv', index=False)

# Display the first few rows of the processed data to verify changes
print(data.head())
print("\nData has been successfully preprocessed and saved.")
