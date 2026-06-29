# 🎵 Spotify Music Recommendation System

A Machine Learning-based Music Recommendation System that suggests songs similar to a user's favorite track using the **K-Nearest Neighbors (KNN)** algorithm. The application is built with **Python** and **Streamlit**, providing an interactive interface for discovering similar songs based on Spotify audio features.

---

## 🚀 Features

* 🎶 Recommend similar songs instantly
* 🔍 Search songs from the Spotify dataset
* 🤖 K-Nearest Neighbors (KNN) recommendation engine
* 📊 Uses Spotify audio features for similarity calculation
* ⚡ Fast predictions using pre-trained models
* 🌐 Interactive web application built with Streamlit

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle

---

## 📂 Project Structure

```text
Spotify-Music-Recommendation/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── music_data.pkl          # Processed dataset
├── music_features.pkl      # Feature matrix
├── music_knn_model.pkl     # Trained KNN model
├── music_scaler.pkl        # Feature scaler
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the **Spotify Tracks Dataset** from Kaggle.

🔗 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

The dataset contains various audio features such as:

* Popularity
* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration
* Explicit

---

## 🧠 How It Works

1. Load the Spotify dataset.
2. Select relevant audio features.
3. Scale the features using StandardScaler.
4. Train a K-Nearest Neighbors (KNN) model.
5. Save the trained model using Pickle.
6. When a user selects a song, the model finds the nearest songs based on feature similarity.
7. Display the recommended songs in the Streamlit interface.

---

## 📈 Machine Learning Workflow

* Data Cleaning
* Feature Selection
* Feature Scaling
* KNN Model Training
* Model Serialization
* Recommendation Generation
* Streamlit Deployment

---

## 🔮 Future Improvements

* Integrate Spotify API for album artwork and song previews
* Recommend songs by mood or genre
* Add artist-based recommendations
* Hybrid recommendation system (Content + Collaborative Filtering)
* Deploy on Streamlit Community Cloud or Render
* Add playlist generation

---

## 📄 License

This project is licensed under the MIT License.
