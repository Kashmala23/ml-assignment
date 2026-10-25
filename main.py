from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
app = FastAPI(title="BankWise Customer Analytics")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise RuntimeError("Models not found. Run: python train_models.py")
    return joblib.load(path)

regression_model = load_model("balance_regression.joblib")
classification_model = load_model("churn_logistic.joblib")
metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

class Customer(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0, le=250000)

def to_frame(customer):
    data = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    if data["Geography"] not in ["France", "Spain", "Germany"]:
        raise HTTPException(422, "Country must be France, Spain, or Germany.")
    if data["Gender"] not in ["Female", "Male"]:
        raise HTTPException(422, "Gender must be Female or Male.")
    return pd.DataFrame([data])

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/metrics")
def get_metrics():
    return metrics

@app.post("/predict/regression")
def predict_balance(customer: Customer):
    prediction = float(regression_model.predict(to_frame(customer))[0])
    return {"predicted_balance": round(max(0, prediction), 2)}

@app.post("/predict/classification")
def predict_churn(customer: Customer):
    row = to_frame(customer)
    prediction = int(classification_model.predict(row)[0])
    probability = float(classification_model.predict_proba(row)[0][1])
    return {
        "prediction": "At risk of leaving" if prediction else "Likely to stay",
        "churn_probability": round(probability * 100, 1),
        "business_advice": (
            "Consider a personal follow-up and review suitable retention options."
            if prediction else
            "Maintain engagement and continue monitoring customer satisfaction."
        )
    }
