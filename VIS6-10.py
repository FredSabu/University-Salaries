import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx


# Load data
data = pd.read_csv('/Users/Mathi/Desktop/final_processed_movies.csv',header=0)
data.columns = data.iloc[0]  # Set the first row as the column header
data = data[1:]  # Remove the first row from the data

''' RUN EACH GRAPH SEPARATELY, copy the stuff above this and then only one graph and run it'''


#6. Box Plot of Movie ratings by Genre
# Convert 'genre_list' from string of list to actual list and explode it
data['genre_list'] = data['genre_list'].apply(eval)
data_exploded = data.explode('genre_list')

# Reset the index after exploding the data
data_exploded = data_exploded.reset_index(drop=True)

# Convert 'vote_average' to numeric for plotting
data_exploded['vote_average'] = pd.to_numeric(data_exploded['vote_average'], errors='coerce')

# Creating a box plot of movie ratings by genre
plt.figure(figsize=(12, 6))
sns.boxplot(x='genre_list', y='vote_average', data=data_exploded)
plt.xticks(rotation=45)  # Rotate genre labels for better visibility
plt.title('Box Plot of Movie Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.show()




#7. Stacked Bar Chart of Movies released per Year by genre
# Convert 'genre_list' from string of list to actual list
data['genre_list'] = data['genre_list'].apply(lambda x: eval(x))

# Exploding 'genre_list' so each genre gets its own row
data_exploded = data.explode('genre_list')

# Creating a new column 'release_year' extracted from 'release_date'
data_exploded['release_year'] = pd.to_datetime(data_exploded['release_date']).dt.year

# Creating a crosstab of the number of movies released per year for each genre
genre_year_data = pd.crosstab(data_exploded['release_year'], data_exploded['genre_list'])

# Plotting directly without calling plt.figure() separately
genre_year_data.plot(kind='bar', stacked=True, figsize=(14, 7), legend=True, title='Stacked Bar Chart of Movies Released per Year by Genre')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



#8. Density Plot of Movie Budgets and Revenues
# Convert 'budget' and 'revenue' to numeric, coercing errors
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')

# Drop NaN values specifically in 'budget' and 'revenue' columns
data = data.dropna(subset=['budget', 'revenue'])

# Filter out entries where budget or revenue is zero, as these do not contribute to meaningful density information
data = data[(data['budget'] > 0) & (data['revenue'] > 0)]

# Applying log transformation to manage wide distribution
data['log_budget'] = np.log(data['budget'])
data['log_revenue'] = np.log(data['revenue'])

# Set up the matplotlib figure
plt.figure(figsize=(24, 6))

# Plot 1: Normal density plot
plt.subplot(1, 2, 1)
sns.kdeplot(data['budget'], shade=True, color="r", label="Budget")
sns.kdeplot(data['revenue'], shade=True, color="b", label="Revenue")
plt.title('Density Plot of Movie Budgets and Revenues')
plt.xlabel('Dollars')
plt.ylabel('Density')
plt.legend()

# Plot 2: Log-density plot
plt.subplot(1, 2, 2)
sns.kdeplot(data['log_budget'], shade=True, color="r", label="Budget")
sns.kdeplot(data['log_revenue'], shade=True, color="b", label="Revenue")
plt.title('Log-Density Plot of Movie Budgets and Revenues')
plt.xlabel('Log Dollars')
plt.ylabel('Density')
plt.legend()

plt.show()



#9. Network Graph of Co-occurrences of Genres in Movies
# Convert 'genre_list' from string of list to actual list
data['genre_list'] = data['genre_list'].apply(eval)

# Initialize graph
G = nx.Graph()

# Add nodes and edges from the 'genre_list'
for index, row in data.iterrows():
    genres = row['genre_list']
    for genre in genres:
        if not G.has_node(genre):
            G.add_node(genre)
    for i in range(len(genres)):
        for j in range(i + 1, len(genres)):
            if G.has_edge(genres[i], genres[j]):
                # Increment the weight by 1 if the edge already exists
                G[genres[i]][genres[j]]['weight'] += 1
            else:
                # Add a new edge with weight 1 if it doesn't exist
                G.add_edge(genres[i], genres[j], weight=1)

# Draw the network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)  # Positions for all nodes
nx.draw_networkx_nodes(G, pos, node_color='blue', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos)
plt.title('Network Graph of Co-occurrence of Genres in Movies')
plt.show()


#10. 3D Scatter Plot of Movies by Popularity, Revenue, and Runtime
# Convert 'popularity', 'revenue', and 'runtime' to numeric, coercing errors
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data['runtime'] = pd.to_numeric(data['runtime'], errors='coerce')

# Drop NaN values
data = data.dropna(subset=['popularity', 'revenue', 'runtime'])

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(data['popularity'], data['revenue'], data['runtime'], c=data['popularity'], cmap='viridis')

# Labels and title
ax.set_xlabel('Popularity')
ax.set_ylabel('Revenue')
ax.set_zlabel('Runtime')
plt.title('3D Scatter Plot of Movies by Popularity, Revenue, and Runtime')

# Color bar
plt.colorbar(sc)

plt.show()


