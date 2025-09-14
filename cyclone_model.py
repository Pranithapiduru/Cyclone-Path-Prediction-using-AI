# cyclone_model.py
import joblib
import pandas as pd

# Load the trained model from the pickle file
_model = joblib.load("cyclone_model.pkl")

def predict(lat: float, lon: float, steps: int = 48) -> pd.DataFrame:
    """
    Generate cyclone path predictions using the trained model.
    Replace the demo logic below with your real preprocessing + _model.predict().
    """
    # 👉 Demo straight line (so app runs even before you connect the real model)
    points = [[lat + i * 0.01, lon + i * 0.01] for i in range(steps)]
    return pd.DataFrame(points, columns=["lat", "lon"])
