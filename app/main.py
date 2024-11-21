from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("app/modelo.pkl")

# Inicializar la API
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Definir un esquema para la entrada
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

# Definir la ruta principal para predicciones
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, feature1: float = Form(...), feature2: float = Form(...), feature3: float = Form(...), feature4: float = Form(...), feature5: float = Form(...)):
    # Convertir los datos a un DataFrame
    input_df = pd.DataFrame([{
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
        "feature5": feature5
    }])
    
    # Realizar predicción
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df).max(axis=1)[0]

    result = "Este usuario pertenece al grupo número 1" if prediction[0] == 1 else "Este usuario pertenece al grupo número 0"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "probability": prob
    })
