from fastapi import FastAPI
from api.schema import user_input
import joblib
from src.features.build_features import column_transformation
import joblib
import pandas as pd

model = joblib.load("models/final_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
median_val = joblib.load("models/median.pkl")

app = FastAPI()

# Load model
model = joblib.load("models/final_model.joblib")

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: user_input):
    
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df = column_transformation(df, median_val)

    # Preprocessing (scaling + OHE)
    df = preprocessor.transform(df)

    # Predict
    prediction = model.predict(df)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df)[0][1]
    else:
        prob = None

    return {
        "prediction": int(prediction),
        "probability": float(prob) if prob is not None else None
    }