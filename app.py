from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="NexusML - Automated Model Lifecycle Management Platform",
    description="Enterprise MLOps platform for Employee Attrition Prediction",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "frontend" / "templates")
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "frontend" / "static")),
    name="static"
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    MonthlyIncome: float = Form(...),
    YearsAtCompany: int = Form(...)
):
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
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
async def health():
    return {
        "status": "healthy",
        "service": "NexusML",
        "version": "1.0.0",
        "model": "loaded"
    }