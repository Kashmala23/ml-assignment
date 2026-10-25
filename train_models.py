"""Train the two models using Churn_Modelling.csv in this folder."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Churn_Modelling.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Churn_Modelling.csv was not found. Copy your dataset into the project folder next to main.py."
    )

df = pd.read_csv(DATA_PATH)
required = ["CreditScore","Geography","Gender","Age","Tenure","Balance",
            "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

features = ["CreditScore","Geography","Gender","Age","Tenure",
            "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
numeric = ["CreditScore","Age","Tenure","NumOfProducts","HasCrCard",
           "IsActiveMember","EstimatedSalary"]
categorical = ["Geography","Gender"]

def preprocessor(scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

# Model 1: Regression predicts the Balance column.
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["Balance"], test_size=0.2, random_state=42
)
regression = Pipeline([
    ("preprocessor", preprocessor()),
    ("model", LinearRegression())
])
regression.fit(X_train, y_train)
balance_predictions = regression.predict(X_test)
metrics = {
    "rows": int(len(df)),
    "regression": {
        "r2": round(float(r2_score(y_test, balance_predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, balance_predictions)), 2)
    }
}
joblib.dump(regression, MODEL_DIR / "balance_regression.joblib")

# Model 2: Classification predicts Exited (0 = stayed, 1 = left).
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["Exited"], test_size=0.2, random_state=42,
    stratify=df["Exited"]
)
classification = Pipeline([
    ("preprocessor", preprocessor(scale=True)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
classification.fit(X_train, y_train)
predictions = classification.predict(X_test)
metrics["classification"] = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "report": classification_report(y_test, predictions, output_dict=True, zero_division=0)
}
joblib.dump(classification, MODEL_DIR / "churn_logistic.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Training complete.")
print(f"Rows: {metrics['rows']}")
print(f"Regression R²: {metrics['regression']['r2']}")
print(f"Regression MAE: {metrics['regression']['mae']}")
print(f"Classification accuracy: {metrics['classification']['accuracy']:.1%}")
print("Saved models in the models folder.")
