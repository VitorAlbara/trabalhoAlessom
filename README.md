API de Predição da Doença de Alzheimer
Uma API para prever a probabilidade de diagnóstico da Doença de Alzheimer com base em dados clínicos e de estilo de vida. 

Descrição
Este projeto consiste numa API desenvolvida com FastAPI que serve um modelo de machine learning treinado para prever a probabilidade de um paciente ter a Doença de Alzheimer. O modelo utiliza um conjunto de dados clínicos e de estilo de vida para fazer as previsões.

A API disponibiliza um endpoint para submeter os dados de um paciente e receber como resposta o diagnóstico previsto (positivo ou negativo) e as probabilidades associadas.

Tecnologias Utilizadas
Python


FastAPI: para a construção da API. 

scikit-learn: para o treino e avaliação do modelo de machine learning.


Pandas: para manipulação e pré-processamento de dados. 


Joblib: para carregar e guardar o modelo treinado. 

Uvicorn: para correr o servidor da API.

Instalação
Clone o repositório:

Bash

git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>
Crie e ative um ambiente virtual (recomendado):

Bash

python -m venv venv
source venv/bin/activate  # No Windows use `venv\Scripts\activate`
Instale as dependências:

Bash

pip install -r requirements.txt
Utilização
1. Treinar o Modelo
Antes de iniciar a API, é necessário treinar o modelo. O script train_model.py encarrega-se de carregar os dados, pré-processá-los, treinar um modelo RandomForestClassifier com GridSearchCV para encontrar os melhores hiperparâmetros e, por fim, guardar o pipeline do modelo treinado no ficheiro alzheimer_model_pipeline.joblib.

Para treinar o modelo, execute o seguinte comando no seu terminal:

Bash

python train_model.py
Certifique-se de que o ficheiro alzheimers_disease_data.csv se encontra no mesmo diretório.

2. Iniciar a API
Após o treino do modelo, o ficheiro alzheimer_model_pipeline.joblib será gerado. Agora, pode iniciar a API com o Uvicorn:

Bash

uvicorn main:app --reload
A API estará disponível em http://127.0.0.1:8000.

Endpoints da API
A API possui os seguintes endpoints:

GET /

Descrição: Endpoint de boas-vindas que exibe uma mensagem e o estado do modelo. 

Resposta de Sucesso (código 200):

JSON

{
  "message": "Bem-vindo à API de Predição da Doença de Alzheimer",
  "model_status": "Modelo carregado com sucesso."
}
POST /predict

Descrição: Recebe os dados de um paciente em formato JSON e retorna a previsão do diagnóstico. 

Corpo do Pedido:

JSON

{
  "Age": 75,
  "Gender": 0,
  "Ethnicity": 0,
  "EducationLevel": 2,
  "BMI": 28.5,
  "Smoking": 0,
  "AlcoholConsumption": 3.5,
  "PhysicalActivity": 4,
  "DietQuality": 6.5,
  "SleepQuality": 7,
  "FamilyHistoryAlzheimers": 1,
  "CardiovascularDisease": 1,
  "Diabetes": 0,
  "Depression": 0,
  "HeadInjury": 0,
  "Hypertension": 1,
  "SystolicBP": 140,
  "DiastolicBP": 90,
  "CholesterolTotal": 200,
  "CholesterolLDL": 130,
  "CholesterolHDL": 50,
  "CholesterolTriglycerides": 150,
  "MMSE": 25,
  "FunctionalAssessment": 7,
  "MemoryComplaints": 1,
  "BehavioralProblems": 0,
  "ADL": 8,
  "Confusion": 0,
  "Disorientation": 0,
  "PersonalityChanges": 0,
  "DifficultyCompletingTasks": 1,
  "Forgetfulness": 1
}
Resposta de Sucesso (código 200):

JSON

{
  "prediction_code": 1,
  "diagnosis": "Alzheimer Positivo",
  "probability_negative": 0.3456,
  "probability_positive": 0.6544
}
Dataset
O modelo foi treinado com o conjunto de dados alzheimers_disease_data.csv, que contém informações demográficas, de estilo de vida, historial clínico e resultados de testes cognitivos de pacientes.
