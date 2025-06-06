

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 📌 1. Carregando Dados Simulados (Exemplo)
df = pd.read_csv('flood.csv', nrows=1000)

print(df.head())

# 📌 2. Separando Features e Target
#X = df[['precipitação_mm', 'nível_rio_m']]
#y = df['alerta_enchente']

# 📌 3. Dividindo os Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 4. Criando e Treinando o Modelo KNN
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# 📌 5. Testando o Modelo
y_pred = modelo.predict(X_test)
precisão = accuracy_score(y_test, y_pred)

print(f'A precisão do modelo KNN é de {precisão:.2%}')

# 📌 6. Simulação de Previsão
entrada = np.array([[95, 5.6]])  # Exemplo: 95mm de chuva e 5.6m de nível do rio
risco = modelo.predict(entrada)

print("⚠️ Alerta de enchente!" if risco[0] == 1 else "✅ Sem risco de enchente.")
