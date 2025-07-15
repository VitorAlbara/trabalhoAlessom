🧠 API de Predição da Doença de Alzheimer
Uma API RESTful de alta performance construída com FastAPI para prever a probabilidade de diagnóstico da Doença de Alzheimer, utilizando um modelo de machine learning treinado com dados clínicos e de estilo de vida.

🚀 Sobre o Projeto
Este projeto disponibiliza uma interface simples e eficaz para interagir com um modelo preditivo de Alzheimer. A API permite que aplicações frontend ou outros serviços submetam dados de um paciente e recebam, em tempo real, um diagnóstico provável e as respetivas probabilidades.

O objetivo é fornecer uma ferramenta que possa auxiliar profissionais de saúde e investigadores, demonstrando o poder da aprendizagem automática na área da saúde.

✨ Como Funciona
O core do projeto é um pipeline de machine learning que automatiza todo o processo, desde o pré-processamento dos dados até à predição.

Carregamento e Limpeza de Dados: O script train_model.py carrega o dataset alzheimers_disease_data.csv.

Pré-processamento:

Colunas irrelevantes (PatientID, DoctorInCharge) são removidas.

Valores em falta são preenchidos com a mediana da respetiva coluna.

As características numéricas são escalonadas utilizando StandardScaler para normalizar a sua distribuição, o que é crucial para o desempenho do modelo.

Treino do Modelo:

Um RandomForestClassifier é utilizado devido à sua robustez e alto desempenho.

GridSearchCV é aplicado para testar diferentes combinações de hiperparâmetros e encontrar a melhor configuração com base na métrica f1_weighted.

Serialização: O pipeline completo (pré-processador + melhor modelo) é guardado no ficheiro alzheimer_model_pipeline.joblib utilizando joblib.


Serviço da API: A aplicação FastAPI (main.py) carrega este ficheiro para disponibilizar as predições através de um endpoint POST. 

🛠️ Tecnologias Utilizadas
Python: Linguagem principal do projeto.
FastAPI :Framework web para a construção da API.
Uvicorn :Servidor ASGI para executar a API.
Scikit-learn :Biblioteca para o treino e avaliação do modelo.
Pandas :Utilizado para a manipulação e análise dos dados.
Joblib :Para carregar e guardar o modelo treinado.

⚙️ Instalação e Execução
Siga estes passos para configurar e executar o projeto localmente.

Pré-requisitos:
Python 3.7+
Pip

Passos
Clone o repositório:

git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>
Crie e ative um ambiente virtual:

python -m venv venv
# No macOS/Linux:
source venv/bin/activate
# No Windows:
.\venv\Scripts\activate
Instale as dependências a partir do requirements.txt:

pip install -r requirements.txt
Treine o modelo:

⚠️ Importante: Este passo é obrigatório na primeira execução. Certifique-se de que o ficheiro alzheimers_disease_data.csv está na raiz do projeto.

python train_model.py
Inicie o servidor da API:

uvicorn main:app --reload
A API estará agora a correr em http://127.0.0.1:8000 e a documentação interativa (Swagger UI) em http://127.0.0.1:8000/docs.

API Endpoints
A API fornece dois endpoints principais.

GET /
Endpoint de boas-vindas para verificar o estado da API e do modelo.

URL: /

Método: GET

Resposta de Sucesso:

JSON

{
  "message": "Bem-vindo à API de Predição da Doença de Alzheimer",
  "model_status": "Modelo carregado com sucesso."
}
POST /predict
Realiza a predição com base nos dados do paciente fornecidos.

URL: /predict

Método: POST

Corpo do Pedido (Exemplo):

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
Resposta de Sucesso:

JSON

{
  "prediction_code": 1,
  "diagnosis": "Alzheimer Positivo",
  "probability_negative": 0.3456,
  "probability_positive": 0.6544
}
📄 Licença
Este projeto está licenciado sob a Licença MIT. Consulte o ficheiro LICENSE para mais detalhes.
