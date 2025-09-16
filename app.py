import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load datasets
# -----------------------------
df_noaa = pd.read_csv("data/noaa_atlantic_hurricane.csv")
df_north_indian = pd.read_csv("data/seasonal_winter_cyclone_frequency.csv")
df_tropical = pd.read_csv("data/tropical_cyclone_tracks.csv")

# -----------------------------
# Prepare training data
# -----------------------------
X = df_tropical.drop(columns=['name', 'lat', 'long', 'status'], errors='ignore')
y = df_tropical[['lat', 'long']]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌪️ Cyclone Path Prediction")
st.write("Enter cyclone details to predict its **Latitude & Longitude**")

user_input = {}
for col in X.columns:
    if X[col].dtype in [np.int64, np.float64]:
        user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    else:
        user_input[col] = st.selectbox(f"{col}", [0, 1], index=0)

input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

if st.button("Predict Location"):
    prediction = model.predict(input_df)
    lat, lon = prediction[0]
    st.success(f"🌍 Predicted Location → Latitude: **{lat:.2f}**, Longitude: **{lon:.2f}**")
