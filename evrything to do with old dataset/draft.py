import pandas as pd

# Reading the Dataset
salaries = pd.read_csv("higher_ed_employee_salaries.csv")

# Convert 'Name' and other suitable columns to 'category' datatype if they have limited unique values
salaries['Name'] = salaries['Name'].astype('category')
# Do the same for other columns that have a limited set of texts such as 'School' or 'Job Description'

# Checking for Missing values only once, storing results in a variable to avoid recalculating
total_rows = salaries.shape[0]
missing_per_column = salaries.isnull().sum()
percentage_missing = (missing_per_column / total_rows) * 100
print("------MISSING------")
print(missing_per_column)
print("\n------PERCENTAGE------")
print(percentage_missing)

# Group by 'Name' and fill missing 'Department' values, avoid sorting if it's not necessary
salaries['Department'] = salaries.groupby('Name')['Department'].transform(lambda x: x.ffill().bfill())

# If you need to sort for the final output, do it now
salaries.sort_values(by=['Name', 'Year'], ascending=[True, False], inplace=True)

# Write to CSV, adjust chunksize if necessary for optimization
salaries.to_csv('preprocessed_salaries.csv, index=False')
print("done")
