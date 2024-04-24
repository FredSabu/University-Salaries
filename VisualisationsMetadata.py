import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import csv

# Load data
data = pd.read_csv('/Users/Mathi/Desktop/movies_metadata.csv')



#1. Histogram of Movie Popularity
# Clean and preprocess data
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data = data.dropna(subset=['popularity'])

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(data['popularity'], bins=50, color='blue', edgecolor='black')
plt.title('Distribution of Movie Popularity')
plt.xlabel('Popularity')
plt.ylabel('Number of Movies')
plt.grid(True)
plt.show()


#2. Scatter Plot of Revenue vs. Budget
# Clean and preprocess
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data = data.dropna(subset=['revenue', 'budget'])
data = data[data['budget'] > 1000]  # Filter out rows where budget is unrealistically low

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['budget'], data['revenue'], alpha=0.5)
plt.title('Movie Revenue vs. Budget')
plt.xlabel('Budget (in billions)')
plt.ylabel('Revenue (in billions)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()


#3. Correlation Heatmap of Numeric Features
# Selecting numerical features
numerical_features = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr = numerical_features.corr()

# Generate a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


#4. Time Series Analysis of Movie Releases and Revenue Over Years
# Convert 'release_date' to datetime
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Extract year from release_date
data['release_year'] = data['release_date'].dt.year

# Group by year and sum revenues
annual_revenue = data.groupby('release_year')['revenue'].sum()

# Plotting
plt.figure(figsize=(12, 6))
annual_revenue.plot(kind='line')
plt.title('Annual Movie Revenues Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.show()



#5. Interactive Bubble Chart of Genres by Popularity and Revenue
# Preprocessing genre data
# Assuming genres column is in JSON-like format and needs to be parsed from a string of JSON objects
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)

# Handling missing values and exploding the genre to normalize the data
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')

# Clean and preprocess numerical data for visualization
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data = data.dropna(subset=['popularity', 'revenue'])

# Group data by genre and compute average popularity and total revenue
genre_stats = data.groupby('genre').agg({
    'popularity': 'mean',
    'revenue': 'sum'
}).reset_index()

# Generate the interactive bubble chart
fig = px.scatter(genre_stats, x="popularity", y="revenue", size="revenue", color="genre",
                 hover_name="genre", size_max=60)
fig.update_layout(title='Genres by Popularity and Revenue',
                  xaxis_title='Average Popularity',
                  yaxis_title='Total Revenue')
fig.show()



#6. Box Plot of Movie ratings by Genre
# Preprocessing genre data from JSON-like strings in 'genres' column
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')

# Preprocessing rating data
data['vote_average'] = pd.to_numeric(data['vote_average'], errors='coerce')
data = data.dropna(subset=['vote_average'])

# Plotting
plt.figure(figsize=(14, 7))
sns.boxplot(x='genre', y='vote_average', data=data)
plt.xticks(rotation=45)
plt.title('Distribution of Movie Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Movie Ratings')
plt.show()




#7. Stacked Bar Chart of Movies released per Year by genre
# Preprocessing genre data
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')

# Convert 'release_date' to datetime and extract year
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year
data = data.dropna(subset=['release_year'])

# Group by year and genre
genre_year_counts = data.groupby(['release_year', 'genre']).size().unstack(fill_value=0)

# Plotting
genre_year_counts.plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title('Number of Movies Released per Year by Genre')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.legend(title='Genre')
plt.show()



#8. Density Plot of Movie Budgets and Revenues
# Preprocessing genre data
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')

# Clean and preprocess numerical data for visualization
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data = data.dropna(subset=['budget', 'revenue'])

# Filter out unrealistic entries (very low budgets and revenues might be incorrect or placeholders)
data_filtered = data[(data['budget'] > 1000) & (data['revenue'] > 1000)]

# Generate the density plot using the correct seaborn kdeplot syntax
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data_filtered, x='budget', y='revenue', cmap="Reds", shade=True, bw_adjust=0.5)
plt.title('Density Plot of Movie Budgets vs. Revenues')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.show()



#9. Network Graph of Co-occurrences of Genres in Movies
G = nx.Graph()

# Add nodes and edges based on genre co-occurrence
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')

for genres in data['genre'].str.split(', ').dropna():
    genres = list(set(genres))  # Remove duplicates in a single movie
    G.add_nodes_from(genres)
    for genre in genres:
        for co_genre in genres:
            if genre != co_genre:
                G.add_edge(genre, co_genre)

pos = nx.spring_layout(G, seed=42)  # positions for all nodes

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        color=[],
    )
)

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (node,)

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color'] += (len(adjacencies[1]),)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network Graph of Genre Co-occurrence',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Python code: NetworkX+Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig.show()


#10. 3D Scatter Plot of Movies by Popularity, Revenue, and Runtime
# Assuming data has necessary columns cleaned and converted to appropriate data types
fig = px.scatter_3d(data, x='popularity', y='revenue', z='runtime', color='vote_average',
                    title='3D Scatter Plot of Movies by Popularity, Revenue, and Runtime')
fig.show()


#11. Animated Time Series Plot of Total Movie Revenues over the Years
# Convert 'release_date' to datetime and extract the year
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year

# Convert 'revenue' to numeric, dropping non-numeric values
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')

# Filter out rows with no revenue or no release year, and exclude zero revenues
data = data[(data['revenue'] > 0) & (data['release_year'].notna())]

# Aggregate revenue by year
annual_revenue = data.groupby('release_year')['revenue'].sum().reset_index()

# Sort by year to ensure the animation flows correctly
annual_revenue = annual_revenue.sort_values('release_year')

# Check the data
print(annual_revenue.head())
print("Earliest year: ", annual_revenue['release_year'].min(), "Latest year: ", annual_revenue['release_year'].max())

# Create an animated plot
fig = px.bar(annual_revenue, x='release_year', y='revenue',
             animation_frame='release_year', animation_group='release_year',
             range_y=[0, annual_revenue['revenue'].max() * 1.1],  # Dynamic y-axis range
             range_x=[annual_revenue['release_year'].min(), annual_revenue['release_year'].max()])  # Dynamic x-axis range

fig.update_layout(title='Total Movie Revenues Over Years',
                  xaxis_title='Year',
                  yaxis_title='Total Revenue',
                  xaxis=dict(tickmode='linear', tick0=annual_revenue['release_year'].min(), dtick=1),
                  showlegend=False)

fig.show()


#12. Radar Chart of Movie Metrics by Genre
def extract_genres(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))
        if genres:
            return ', '.join([g['name'] for g in genres])
        return None
    except:
        return None

data['genre'] = data['genres'].apply(extract_genres)
data = data.dropna(subset=['genre'])
data = data.assign(genre=data['genre'].str.split(', ')).explode('genre')


# Convert budget and revenue to numeric, removing any non-numeric values
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data['vote_average'] = pd.to_numeric(data['vote_average'], errors='coerce')

# Filter out unrealistic or placeholder entries
data_filtered = data[(data['budget'] > 1000) & (data['revenue'] > 1000)]

# Group data by genre and calculate mean values for several metrics
genre_metrics = data_filtered.groupby('genre').agg({
    'budget': 'mean',
    'revenue': 'mean',
    'popularity': 'mean',
    'vote_average': 'mean'
}).reset_index()

# Select a few genres for visualization
selected_genres = genre_metrics['genre'].sample(n=5)  # Randomly pick 5 genres
data_selected = genre_metrics[genre_metrics['genre'].isin(selected_genres)]

# Create radar chart
fig = go.Figure()

for genre in data_selected['genre']:
    fig.add_trace(go.Scatterpolar(
        r=[
            data_selected[data_selected['genre'] == genre]['budget'].values[0],
            data_selected[data_selected['genre'] == genre]['revenue'].values[0],
            data_selected[data_selected['genre'] == genre]['popularity'].values[0],
            data_selected[data_selected['genre'] == genre]['vote_average'].values[0]
        ],
        theta=['budget', 'revenue', 'popularity', 'rating'],
        fill='toself',
        name=genre
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, max(genre_metrics['budget'].max(), genre_metrics['revenue'].max(), genre_metrics['popularity'].max(), genre_metrics['vote_average'].max())]
    )),
  title='Radar Chart of Movie Metrics by Genre',
  showlegend=True
)

fig.show()
