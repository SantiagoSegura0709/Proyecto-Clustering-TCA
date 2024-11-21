from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("modelo.pkl")

# Inicializar la API
app = FastAPI()

# Definir un esquema para la entrada
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Definir la ruta principal para predicciones
@app.post("/predict")
def predict(data: InputData):
    # Convertir los datos a un DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Realizar predicción
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df).max(axis=1)[0]

    return {
        "prediction": int(prediction[0]),
        "probability": prob
    }
