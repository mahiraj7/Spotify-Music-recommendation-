import streamlit as st
import pandas as pd
import joblib

model = joblib.load("music_knn_model.pkl")
scaler = joblib.load("music_scaler.pkl")
data = joblib.load("music_data.pkl")
features = joblib.load("music_features.pkl")

st.title("Music Recommendation System using KNN")

st.write("Select a song and get similar song recommendations.")

song_column = "track_name"

selected_song = st.selectbox(
    "Choose a Song",
    data[song_column].values
)

def recommend_song(track_name):
    song_row = data[data[song_column] == track_name]

    song_features = song_row[features]
    song_scaled = scaler.transform(song_features)
    distances, indices = model.kneighbors(song_scaled)

    recommendations = data.iloc[indices[0]]
    recommendations = recommendations[recommendations[song_column] != track_name]

    return recommendations.head(5)

if st.button("Recommend Songs"):
    result = recommend_song(selected_song)
    st.subheader("Recommended Songs")

    for index, row in result.iterrows():
        st.write("Song:", row["track_name"])
        st.write("Artist:", row.get("artist", "Unknown"))
        st.write("Genre:", row.get("genre", "Unknown"))
        st.write("---")
