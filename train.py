import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("spotify_music.csv")
print(data.head())

print("Dataset Shape :", data.shape)

print("Missing Values:",)
print(data.isnull().sum())

data = data.dropna()

features = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "tempo"
]

X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

model.fit(X_scaled)

joblib.dump(model, "music_knn_model.pkl")
joblib.dump(scaler, "music_scaler.pkl")
joblib.dump(data, "music_data.pkl")
joblib.dump(features, "music_features.pkl")

print("Music recommendation model saved successfully!")
