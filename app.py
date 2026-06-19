from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.src.pipeline.prediction_pipeline import PredictionPipeline

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