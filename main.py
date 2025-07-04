from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
# Importação necessária para o CORS
from fastapi.middleware.cors import CORSMiddleware

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="API de Predição da Doença de Alzheimer",
    description="Uma API para prever a probabilidade de diagnóstico da Doença de Alzheimer com base em dados clínicos e de estilo de vida.",
    version="1.0.0"
)

# --- INÍCIO DA CORREÇÃO DE CORS ---

# Lista de origens que têm permissão para aceder à API.
# Adicione aqui o endereço do seu frontend se for diferente.
origins = [
    "http://localhost",
    "http://localhost:3000",  # Endereço padrão para Next.js/React
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Permite as origens especificadas
    allow_credentials=True,
    allow_methods=["*"],        # Permite todos os métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],        # Permite todos os cabeçalhos
)

# --- FIM DA CORREÇÃO DE CORS ---


# Carrega o pipeline de modelo treinado
try:
    model = joblib.load('alzheimer_model_pipeline.joblib')
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    model = None
    print("ERRO: Arquivo 'alzheimer_model_pipeline.joblib' não encontrado.")
    print("Execute o script 'train_model.py' primeiro para treinar e salvar o modelo.")


# Define a estrutura de dados de entrada usando Pydantic
class PatientData(BaseModel):
    Age: int = Field(..., example=75)
    Gender: int = Field(..., example=0)
    Ethnicity: int = Field(..., example=0)
    EducationLevel: int = Field(..., example=2)
    BMI: float = Field(..., example=28.5)
    Smoking: int = Field(..., example=0)
    AlcoholConsumption: float = Field(..., example=3.5)
    PhysicalActivity: float = Field(..., example=4)
    DietQuality: float = Field(..., example=6.5)
    SleepQuality: float = Field(..., example=7)
    FamilyHistoryAlzheimers: int = Field(..., example=1)
    CardiovascularDisease: int = Field(..., example=1)
    Diabetes: int = Field(..., example=0)
    Depression: int = Field(..., example=0)
    HeadInjury: int = Field(..., example=0)
    Hypertension: int = Field(..., example=1)
    SystolicBP: int = Field(..., example=140)
    DiastolicBP: int = Field(..., example=90)
    CholesterolTotal: float = Field(..., example=200)
    CholesterolLDL: float = Field(..., example=130)
    CholesterolHDL: float = Field(..., example=50)
    CholesterolTriglycerides: float = Field(..., example=150)
    MMSE: float = Field(..., example=25)
    FunctionalAssessment: float = Field(..., example=7)
    MemoryComplaints: int = Field(..., example=1)
    BehavioralProblems: int = Field(..., example=0)
    ADL: float = Field(..., example=8)
    Confusion: int = Field(..., example=0)
    Disorientation: int = Field(..., example=0)
    PersonalityChanges: int = Field(..., example=0)
    DifficultyCompletingTasks: int = Field(..., example=1)
    Forgetfulness: int = Field(..., example=1)

# Endpoint de boas-vindas
@app.get("/", summary="Endpoint de Boas-Vindas", description="Exibe uma mensagem de boas-vindas e status da API.")
def read_root():
    return {
        "message": "Bem-vindo à API de Predição da Doença de Alzheimer",
        "model_status": "Modelo carregado com sucesso." if model else "Modelo não carregado. Execute train_model.py"
    }

# Endpoint de predição
@app.post("/predict", summary="Realiza a Predição de Alzheimer", description="Recebe os dados de um paciente e retorna o diagnóstico previsto.")
def predict(data: PatientData):
    if not model:
        raise HTTPException(status_code=503, detail="Modelo não está disponível. Treine o modelo primeiro.")

    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        diagnosis = "Alzheimer Positivo" if prediction[0] == 1 else "Alzheimer Negativo"
        
        return {
            "prediction_code": int(prediction[0]),
            "diagnosis": diagnosis,
            "probability_negative": round(prediction_proba[0][0], 4),
            "probability_positive": round(prediction_proba[0][1], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a predição: {str(e)}")