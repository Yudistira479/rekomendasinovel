import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data

def load_data():
    df = pd.read_csv("novels.csv")
    return df

# Encode categorical features
def preprocess_data(df):
    df = df.copy()
    label_encoders = {}
    for col in ["genre", "author", "status"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

# Train Random Forest model for recommendations
def train_model(df):
    features = ["genre", "rating", "views", "likes", "chapter_count"]
    target = "popularity"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Make predictions based on user input
def get_recommendations(model, df, label_encoders, user_input):
    input_df = pd.DataFrame([user_input])
    for col in ["genre", "author", "status"]:
        le = label_encoders[col]
        input_df[col] = le.transform([input_df[col][0]])
    features = ["genre", "rating", "views", "likes", "chapter_count"]
    predictions = model.predict(df[features])
    df["predicted_popularity"] = predictions
    top_novels = df.sort_values(by="predicted_popularity", ascending=False).head(10)
    return top_novels

# Save search history
def save_history(novels):
    if os.path.exists("history.pkl"):
        with open("history.pkl", "rb") as f:
            history = pickle.load(f)
    else:
        history = []
    history.append(novels)
    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)

# Load search history
def load_history():
    if os.path.exists("history.pkl"):
        with open("history.pkl", "rb") as f:
            return pickle.load(f)
    return []

# Main App
st.set_page_config(page_title="Sistem Rekomendasi Novel", layout="wide")
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Rekomendasi Novel", "Top 10 Novel"])

df = load_data()
processed_df, label_encoders = preprocess_data(df)
model = train_model(processed_df)

if page == "Home":
    st.title("Beranda - Riwayat Pencarian Rekomendasi Novel")
    history = load_history()
    if history:
        for i, rec in enumerate(history[::-1]):
            st.subheader(f"Hasil Pencarian #{len(history)-i}")
            st.dataframe(rec)
    else:
        st.info("Belum ada riwayat pencarian.")

elif page == "Rekomendasi Novel":
    st.title("Rekomendasi Novel Berdasarkan Preferensi Anda")
    genre = st.selectbox("Pilih Genre", df["genre"].unique())
    author = st.selectbox("Pilih Penulis", df["author"].unique())
    status = st.selectbox("Pilih Status", df["status"].unique())
    rating = st.slider("Rating", 0.0, 5.0, 3.0, 0.1)
    views = st.number_input("Jumlah Views", min_value=0, value=1000)
    likes = st.number_input("Jumlah Likes", min_value=0, value=100)
    chapters = st.number_input("Jumlah Chapter", min_value=1, value=10)

    if st.button("Dapatkan Rekomendasi"):
        user_input = {
            "genre": genre,
            "author": author,
            "status": status,
            "rating": rating,
            "views": views,
            "likes": likes,
            "chapter_count": chapters,
        }
        recommendations = get_recommendations(model, processed_df.copy(), label_encoders, user_input)
        st.subheader("10 Rekomendasi Novel Terbaik Untuk Anda")
        st.dataframe(recommendations)
        save_history(recommendations)

elif page == "Top 10 Novel":
    st.title("Top 10 Novel Berdasarkan Rating dan Genre")

    top_rated = df.sort_values(by="rating", ascending=False).head(10)
    st.subheader("10 Novel dengan Rating Tertinggi")
    st.dataframe(top_rated)

    genre_counts = df["genre"].value_counts().head(10).index
    top_genre_df = df[df["genre"].isin(genre_counts)]
    st.subheader("10 Novel Genre Terbaik (Genre Terpopuler)")
    st.dataframe(top_genre_df.head(10))
