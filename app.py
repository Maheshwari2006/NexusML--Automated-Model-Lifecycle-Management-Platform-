import pandas as pd
import shap
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.src.pipeline.prediction_pipeline import PredictionPipeline
from backend.src.pipeline.explanation_pipeline import ExplanationPipeline
app = FastAPI()

templates = Jinja2Templates(
    directory="frontend/templates"
)

app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.post("/predict")
def predict(
        request: Request,
        Age: int = Form(...),
        MonthlyIncome: int = Form(...),
        YearsAtCompany: int = Form(...)
):

    predictor = PredictionPipeline()

    result = predictor.predict(
        Age,
        MonthlyIncome,
        YearsAtCompany
    )

    return {
        "Prediction": result
    }
ExplanationPipeline().run_pipeline()

print("Explainable AI Completed")

model = joblib.load(
    "backend/artifacts/model.pkl"
)

explainer = shap.TreeExplainer(model)
@app.post("/explain")
def explain_prediction(
        Age: int = Form(...),
        MonthlyIncome: int = Form(...),
        YearsAtCompany: int = Form(...)
):

    data = pd.DataFrame({

        "Age":[Age],

        "MonthlyIncome":[MonthlyIncome],

        "YearsAtCompany":[YearsAtCompany]

    })

    prediction = model.predict(data)

    shap_values = explainer.shap_values(data)

    feature_importance = {}

    for i, feature in enumerate(data.columns):

        if isinstance(shap_values, list):

            value = float(shap_values[1][0][i])

        else:

            value = float(shap_values[0][i])

        feature_importance[feature] = round(value,4)

    return {

        "Prediction":

        "Employee Likely To Leave"

        if prediction[0]==1

        else

        "Employee Likely To Stay",

        "Feature Importance":feature_importance

    }

import joblib

try:
    model = joblib.load("backend/artifacts/model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)