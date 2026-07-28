from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="NexusML - Automated Model Lifecycle Management Platform",
    version="1.0.0",
    description="Employee Attrition Prediction using Machine Learning"
)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home():
    return {
        "message": "Welcome to NexusML",
        "status": "Running Successfully"
    }


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    MonthlyIncome: float = Form(...),
    YearsAtCompany: int = Form(...)
):
    prediction = PredictionPipeline().predict(
        Age,
        MonthlyIncome,
        YearsAtCompany
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction
        }
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }