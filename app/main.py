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
    id_estatus_reservaciones: float
    days_to_arrival: float
    late_booking: float
    reservation_day_of_week: float
    id_paquete: float
    ID_Segmento_Comp: float
    week_of_year: float
    h_tfa_total: float
    month: float
    h_num_noc: float

# Definir la ruta principal para predicciones
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  id_estatus_reservaciones: float = Form(...), 
                  days_to_arrival: float = Form(...), 
                  late_booking: float = Form(...), 
                  reservation_day_of_week: float = Form(...), 
                  id_paquete: float = Form(...), 
                  ID_Segmento_Comp: float = Form(...), 
                  week_of_year: float = Form(...), 
                  h_tfa_total: float = Form(...), 
                  month: float = Form(...), 
                  h_num_noc: float = Form(...)):
    # Convertir los datos a un DataFrame
    input_df = pd.DataFrame([{
        "id_estatus_reservaciones": id_estatus_reservaciones,
        "days_to_arrival": days_to_arrival,
        "late_booking": late_booking,
        "reservation_day_of_week": reservation_day_of_week,
        "id_paquete": id_paquete,
        "ID_Segmento_Comp": ID_Segmento_Comp,
        "week_of_year": week_of_year,
        "h_tfa_total": h_tfa_total,
        "month": month,
        "h_num_noc": h_num_noc
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

