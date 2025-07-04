import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

def train_and_save_model():
    """
    Esta função executa o pipeline completo de treinamento:
    1. Carrega os dados.
    2. Realiza o pré-processamento.
    3. Treina o melhor modelo usando GridSearchCV.
    4. Salva o pipeline treinado (pré-processador + modelo) em um arquivo.
    """
    print("Iniciando o processo de treinamento...")

    # 1. Carregar os dados
    # Certifique-se de que o arquivo CSV está no mesmo diretório ou forneça o caminho correto.
    try:
        # O seu notebook usa um caminho do Kaggle, aqui ajustamos para um ficheiro local.
        df = pd.read_csv('alzheimers_disease_data.csv')
        print("Dados carregados com sucesso.")
    except FileNotFoundError:
        print("Erro: 'alzheimers_disease_data.csv' não encontrado.")
        print("Por favor, baixe o dataset e coloque-o na mesma pasta do script.")
        return

    # 2. Pré-processamento
    # Removendo colunas que não serão usadas no modelo
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

    # Lidando com valores ausentes (se houver) - preenchendo com a mediana
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    print("Pré-processamento inicial concluído.")

    # Separando features (X) e alvo (y)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Identificando colunas numéricas para escalonamento
    # Colunas com mais de 10 valores únicos são consideradas numéricas para escalonamento.
    numerical_cols = [col for col in X.columns if df[col].nunique() > 10]
    print(f"Colunas numéricas a serem escalonadas: {numerical_cols}")

    # 3. Criando o Pipeline de Pré-processamento e Modelagem
    # O ColumnTransformer aplica transformações a colunas específicas.
    # Aqui, aplicamos StandardScaler apenas às colunas numéricas.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'  # Mantém as outras colunas (categóricas/binárias) como estão
    )

    # O Pipeline encadeia o pré-processador com o modelo.
    # Isto garante que os novos dados para previsão passem pelas mesmas etapas.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Dividindo os dados para validação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Dados divididos em conjuntos de treino e teste.")

    # Definindo os hiperparâmetros para o GridSearchCV
    # Focando no RandomForestClassifier, que geralmente tem um bom desempenho.
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    # Usando GridSearchCV para encontrar os melhores hiperparâmetros
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
    print("Iniciando a busca por hiperparâmetros com GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Exibindo os melhores parâmetros e o score
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor score F1 (ponderado) em validação cruzada: {grid_search.best_score_:.4f}")

    # Avaliando o melhor modelo no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nRelatório de Classificação no Conjunto de Teste:")
    print(classification_report(y_test, y_pred))

    # 4. Salvando o pipeline treinado
    joblib.dump(best_model, 'alzheimer_model_pipeline.joblib')
    print("\nModelo treinado e salvo com sucesso como 'alzheimer_model_pipeline.joblib'")

if __name__ == '__main__':
    train_and_save_model()
