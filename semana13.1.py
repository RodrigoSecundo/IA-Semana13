"""
Exercício 1: Prevendo o peso de uma pessoa pela altura

Descrição
Implemente em Python (usando pandas e scikit-learn) um modelo de regressão linear simples que, dado um conjunto de dados com altura (em cm) e peso (em kg), estime o peso de uma pessoa.

Passos obrigatórios

Crie manualmente um pequeno dataset (mínimo 10 registros) com colunas altura e peso.
Divida os dados em variáveis X (altura) e y (peso).
Ajuste o modelo usando LinearRegression do scikit-learn.
Exiba os coeficientes a (intercepto) e b (inclinação) da equação y = a + b⋅x
Calcule o R² do modelo.
Preveja o peso esperado para uma pessoa de 170 cm.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    'altura': [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    'peso': [50, 53, 56, 60, 65, 68, 72, 77, 82, 85]
}
df = pd.DataFrame(data)

X = df[['altura']]  # deve ser 2D
y = df['peso']

model = LinearRegression()
model.fit(X, y)

a = model.intercept_
b = model.coef_[0]
print(f"Equação: peso = {a:.2f} + {b:.2f} * altura")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.4f}")

peso_170 = model.predict([[170]])
print(f"Peso previsto para 170 cm: {peso_170[0]:.2f} kg")
