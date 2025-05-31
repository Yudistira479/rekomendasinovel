import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("novels.csv")
    required_cols = ["title", "authors", "genres", "score", "popularty", "status"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut hilang di dataset: {missing_cols}")
        st.stop()
    return df[required_cols].dropna()

# Encode categorical features
def preprocess_data(df):
    df = df.copy()
    label_encoders = {}
    for col in ["genres", "authors", "status"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

# Train Random Forest model
def train_model(df):
    features = ["genres", "score", "status"]
    target = "popularty"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Make predictions
def get_recommendations(model, df, label_encoders, user_input):
    input_df = pd.DataFrame([user_input])
    for col in ["genres", "authors", "status"]:
        le = label_encoders[col]
        input_df[col] = le.transform([input_df[col][0]])
    features = ["genres", "scored", "status"]
    predictions = model.predict(df[features])
    df["predicted_popularty"] = predictions
    top_novels = df.sort_values(by="predicted_popularty", ascending=False).head(10)
    return top_novels

# Setup Streamlit App
st.set_page_config(page_title="Sistem Rekomendasi Novel", layout="wide")
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Rekomendasi Novel", "Top 10 Novel"])

df = load_data()
processed_df, label_encoders = preprocess_data(df)
model = train_model(processed_df)

# Inisialisasi session_state untuk riwayat
if "history" not in st.session_state:
    st.session_state.history = []

if page == "Home":
    st.title("Beranda - Riwayat Pencarian Rekomendasi Novel")
    if st.session_state.history:
        for i, rec in enumerate(reversed(st.session_state.history)):
            st.subheader(f"Hasil Pencarian #{len(st.session_state.history) - i}")
            st.dataframe(rec[["title", "authors", "genres", "score", "popularty", "status"]])
    else:
        st.info("Belum ada riwayat pencarian.")

elif page == "Rekomendasi Novel":
    st.title("Rekomendasi Novel Berdasarkan Preferensi Anda")
    genre = st.selectbox("Pilih Genre", df["genres"].unique())
    author = st.selectbox("Pilih Penulis", df["authors"].unique())
    status = st.selectbox("Pilih Status", df["status"].unique())
    score = st.slider("Skor Rating", 0.0, 5.0, 3.0, 0.1)

    if st.button("Dapatkan Rekomendasi"):
        user_input = {
            "genres": genre,
            "authors": author,
            "status": status,
            "score": score,
        }
        recommendations = get_recommendations(model, processed_df.copy(), label_encoders, user_input)
        st.subheader("10 Rekomendasi Novel Terbaik Untuk Anda")
        st.dataframe(recommendations[["title", "authors", "genres", "scored", "popularty", "status"]])
        st.session_state.history.append(recommendations[["title", "authors", "genres", "score", "popularty", "status"]])

elif page == "Top 10 Novel":
    st.title("Top 10 Novel Berdasarkan Rating dan Genre")

    top_rated = df.sort_values(by="scored", ascending=False).head(10)
    st.subheader("10 Novel dengan Skor Tertinggi")
    st.dataframe(top_rated[["title", "authors", "genres", "score", "popularty", "status"]])

    top_genres = df["genres"].value_counts().head(10).index
    top_genre_df = df[df["genres"].isin(top_genres)]
    st.subheader("10 Novel Genre Terbaik (Genre Terpopuler)")
    st.dataframe(top_genre_df.head(10)[["title", "authors", "genres", "score", "popularty", "status"]])
