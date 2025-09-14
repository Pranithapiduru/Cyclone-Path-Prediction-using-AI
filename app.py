import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit page config ---
st.set_page_config(page_title="Cyclone Path Prediction", page_icon="🌪", layout="centered")

st.title("🌪 Cyclone Path Prediction App")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cyclone_model.pkl")
WINTER_CSV = os.path.join(BASE_DIR, "seasonal_winter_cyclone_frequency.csv")
TRACKS_CSV = os.path.join(BASE_DIR, "tropical_cyclone_tracks.csv")


# ---------- Load or Train Model ----------
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Train a dummy model if no .pkl exists
    st.warning("Model file not found. Training a simple model using available data...")

    try:
        winter_df = pd.read_csv(WINTER_CSV)
        tracks_df = pd.read_csv(TRACKS_CSV)
        df = pd.concat([winter_df, tracks_df], axis=0, ignore_index=True)
    except Exception as e:
        st.error(f"Could not read datasets: {e}")
        return None

    # Minimal preprocessing: drop non-numeric cols
    df = df.select_dtypes(include=["float64", "int64"]).dropna()
    if df.shape[1] < 2:
        st.error("Not enough numeric columns to train.")
        return None

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    st.success("Model trained and saved successfully.")
    return model


model = get_model()

if model is None:
    st.stop()


# ---------- Sidebar Inputs ----------
st.sidebar.header("Input Parameters")

# Replace these with relevant features from your dataset
param1 = st.sidebar.number_input("Parameter 1", value=0.0)
param2 = st.sidebar.number_input("Parameter 2", value=0.0)
param3 = st.sidebar.number_input("Parameter 3", value=0.0)

if st.sidebar.button("Predict Cyclone Path"):
    try:
        input_df = pd.DataFrame([[param1, param2, param3]])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Cyclone Path / Intensity: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- Footer ----------
st.caption("Make sure your .csv files and model are in the same folder as app.py")



