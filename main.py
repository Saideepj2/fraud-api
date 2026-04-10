from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Claim(BaseModel):
    insured_zip: int
    auto_make: str
    incident_type: str
    collision_type: str
    incident_severity: str
    incident_state: str
    total_claim_amount: float
    claim_delay_days: int
    customer_age: int
    vehicle_age: int

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(data: Claim):

    # Step 1: Convert input to dict
    input_dict = data.dict()

    # Step 2: Add derived feature
    input_dict["claim_amount_log"] = np.log1p(input_dict["total_claim_amount"])

    # Step 3: Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Step 4: One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Step 5: Align with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 6: Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    print("DEBUG INPUT:")
    print(input_encoded.head())

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }

