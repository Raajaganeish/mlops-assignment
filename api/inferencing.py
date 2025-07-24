from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
 
# Load the trained model (ensure this path matches your model location)
model = joblib.load("models/best_model.pkl")
 
# Initialize FastAPI app
app = FastAPI(title="Housing Price Predictor")
 
# Define input data schema
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
 
@app.get("/")
def root():
    return {"message": "Welcome to the Housing Price Prediction API"}
 
@app.post("/predict")
def predict(features: HousingFeatures):
    input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                            features.AveBedrms, features.Population, features.AveOccup,
                            features.Latitude, features.Longitude]])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}