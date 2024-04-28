import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json

# Load data
data = pd.read_csv('/Users/Mathi/Desktop/final_processed_movies.csv',header=1)



#1. Histogram of Movie Popularity
# Ensure 'popularity' is treated as a numeric column
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')

# Plotting the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['popularity'].dropna(), bins=30, kde=False)  # Drop NA values for safety
plt.title('Distribution of Movie Popularity')
plt.xlabel('Popularity Score')
plt.ylabel('Number of Movies')
plt.grid(True)
plt.show()



#2. Scatter Plot of Revenue vs. Budget
plt.figure(figsize=(10, 6))
sns.scatterplot(x='budget', y='revenue', data=data)
plt.xscale('log')
plt.yscale('log')
plt.title('Scatter Plot of Revenue vs Budget')
plt.xlabel('Budget (log scale)')
plt.ylabel('Revenue (log scale)')
plt.grid(True)
plt.show()



#3. Correlation Heatmap of Numeric Features
corr_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()



#4. Time Series Analysis of Movie Releases and Revenue Over Years
data['release_year'] = pd.to_datetime(data['release_date']).dt.year
yearly_data = data.groupby('release_year').agg({'title': 'count', 'revenue': 'sum'})

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.lineplot(x=yearly_data.index, y=yearly_data['title'])
plt.title('Number of Movies Released Per Year')
plt.ylabel('Number of Movies')
plt.xlabel('Year')

plt.subplot(1, 2, 2)
sns.lineplot(x=yearly_data.index, y=yearly_data['revenue'])
plt.title('Total Revenue by Year')
plt.ylabel('Total Revenue')
plt.xlabel('Year')
plt.tight_layout()
plt.show()




#5. Interactive Bubble Chart of Genres by Popularity and Revenue
data['genre_list'] = data['genre_list'].str.strip('[]').str.replace("'", "").str.split(', ')
genre_data_exploded = data.explode('genre_list')
genre_data_exploded['popularity'] = genre_data_exploded['popularity'].astype(float)
genre_data_exploded['revenue'] = genre_data_exploded['revenue'].astype(float)

# Grouping data by genre and calculating average popularity and revenue
genre_summary = genre_data_exploded.groupby('genre_list').agg({'popularity': 'mean', 'revenue': 'mean'}).reset_index()

# Creating the interactive bubble chart
fig = px.scatter(genre_summary, x="popularity", y="revenue", size="revenue", color="genre_list", hover_name="genre_list",
                 log_x=True, size_max=60, title="Interactive Bubble Chart of Genres by Popularity and Revenue")
fig.show()
