# ==========================================================
# SPOTIFY SONG GENRE SEGMENTATION & RECOMMENDATION SYSTEM
# ==========================================================

# --------------------------
# IMPORT LIBRARIES
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sns.set_style("whitegrid")

# --------------------------
# LOAD DATASET
# --------------------------
df = pd.read_csv("spotify_songs.csv")

print("="*50)
print("DATASET INFORMATION")
print("="*50)

print("Shape :", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------
# DATA PREPROCESSING
# --------------------------

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nShape After Cleaning :", df.shape)

# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

# 1 Popularity Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['track_popularity'], bins=30, kde=True)
plt.title("Track Popularity Distribution")
plt.show()

# 2 Genre Distribution

plt.figure(figsize=(10,5))
sns.countplot(
    x='playlist_genre',
    data=df,
    order=df['playlist_genre'].value_counts().index
)
plt.xticks(rotation=45)
plt.title("Playlist Genre Distribution")
plt.show()

# 3 Top Playlist Names

plt.figure(figsize=(12,6))
df['playlist_name'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Playlist Names")
plt.ylabel("Number of Songs")
plt.show()

# 4 Popularity By Genre

plt.figure(figsize=(10,5))
sns.boxplot(
    x='playlist_genre',
    y='track_popularity',
    data=df
)
plt.xticks(rotation=45)
plt.title("Popularity by Genre")
plt.show()

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

plt.figure(figsize=(14,10))

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=False
)

plt.title("Correlation Matrix")
plt.show()

# ==========================================================
# AUDIO FEATURES
# ==========================================================

features = [
    'danceability',
    'energy',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

# Feature distributions

df[features].hist(
    figsize=(14,10),
    bins=20
)

plt.suptitle("Audio Feature Distributions")
plt.show()

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[features])

# ==========================================================
# ELBOW METHOD
# ==========================================================

wcss = []

for i in range(1,11):
    
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    
    model.fit(X_scaled)
    
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ==========================================================
# K-MEANS CLUSTERING
# ==========================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts")
print(df["Cluster"].value_counts())

# ==========================================================
# PCA VISUALIZATION
# ==========================================================

pca = PCA(n_components=2)

pca_features = pca.fit_transform(X_scaled)

df["PCA1"] = pca_features[:,0]
df["PCA2"] = pca_features[:,1]

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set1"
)

plt.title("Spotify Song Clusters")
plt.show()

# ==========================================================
# CLUSTERS BY PLAYLIST GENRE
# ==========================================================

plt.figure(figsize=(10,6))

sns.countplot(
    x='playlist_genre',
    hue='Cluster',
    data=df
)

plt.xticks(rotation=45)

plt.title("Clusters Across Genres")

plt.show()

# ==========================================================
# CLUSTERS BY PLAYLIST NAME
# ==========================================================

top_playlists = df['playlist_name'].value_counts().head(10).index

playlist_df = df[
    df['playlist_name'].isin(top_playlists)
]

plt.figure(figsize=(12,6))

sns.countplot(
    y='playlist_name',
    hue='Cluster',
    data=playlist_df
)

plt.title("Clusters Across Top Playlists")

plt.show()

# ==========================================================
# RECOMMENDATION SYSTEM
# ==========================================================

similarity_matrix = cosine_similarity(X_scaled)

def recommend_song(song_name, n=5):

    song_name = song_name.lower()

    result = df[
        df['track_name'].str.lower()==song_name
    ]

    if len(result)==0:
        print("Song Not Found")
        return

    idx = result.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )[1:n+1]

    recommendations = []

    for i in similarity_scores:

        recommendations.append([
            df.iloc[i[0]]['track_name'],
            df.iloc[i[0]]['track_artist'],
            round(i[1],3)
        ])

    rec_df = pd.DataFrame(
        recommendations,
        columns=[
            "Track",
            "Artist",
            "Similarity Score"
        ]
    )

    return rec_df

# ==========================================================
# SAMPLE TEST
# ==========================================================

sample_song = df.iloc[0]['track_name']

print("\nSample Song:")
print(sample_song)

print("\nRecommended Songs:")

print(recommend_song(sample_song))

# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary")
print(cluster_summary)

print("\nPROJECT COMPLETED SUCCESSFULLY")