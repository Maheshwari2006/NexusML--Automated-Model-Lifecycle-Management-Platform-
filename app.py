
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.src.pipeline.prediction_pipeline import PredictionPipeline
from backend.src.pipeline.model_registry_api import (
    router as model_registry_router,
)
from backend.src.monitoring.monitoring_service import (
    log_prediction,
    get_monitoring_summary,
    detect_data_drift,
    save_monitoring_report,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="NexusML - Automated Model Lifecycle Management Platform",
    description=(
        "Enterprise MLOps platform for "
        "Employee Attrition Prediction"
    ),
    version="1.0.0",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "frontend" / "templates")
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "frontend" / "static")),
    name="static",
)

app.include_router(model_registry_router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None},
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    MonthlyIncome: float = Form(...),
    YearsAtCompany: int = Form(...),
):
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        Age,
        MonthlyIncome,
        YearsAtCompany,
    )

    log_prediction(
        age=Age,
        income=MonthlyIncome,
        years=YearsAtCompany,
        prediction=prediction,
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": prediction},
    )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "NexusML",
        "version": "1.0.0",
        "model": "loaded",
    }

@app.get("/monitoring/summary")
async def monitoring_summary():
    return get_monitoring_summary()

@app.get("/monitoring/drift")
async def monitoring_drift():
    return detect_data_drift()

@app.post("/monitoring/report")
async def generate_monitoring_report():
    return save_monitoring_report()
