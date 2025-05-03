"""
Exercício 2: Prevendo a octanagem da gasolina pelo percentual de aditivo

Descrição
Baseado no exemplo 2 do slide, implemente um modelo que relacione o percentual de aditivo (X) com a octanagem da gasolina (Y).

Passos obrigatórios

Crie manualmente um dataset com os valores:
Percentual aditivo: [1, 2, 3, 4, 5, 6]
Octanagem: [87, 88, 90, 92, 94, 95] (valores fictícios para teste)
Ajuste o modelo usando LinearRegression.
Exiba a equação da reta.
Calcule o R².
Estime a octanagem para 5,5% de aditivo.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dados = {
    'aditivo': [1, 2, 3, 4, 5, 6],
    'octanagem': [87, 88, 90, 92, 94, 95]
}
df2 = pd.DataFrame(dados)

X2 = df2[['aditivo']]
y2 = df2['octanagem']

modelo2 = LinearRegression()
modelo2.fit(X2, y2)

a2 = modelo2.intercept_
b2 = modelo2.coef_[0]
print(f"Equação: octanagem = {a2:.2f} + {b2:.2f} * aditivo")

y2_pred = modelo2.predict(X2)
r2_2 = r2_score(y2, y2_pred)
print(f"R²: {r2_2:.4f}")

octanagem_55 = modelo2.predict([[5.5]])
print(f"Octanagem prevista para 5,5% de aditivo: {octanagem_55[0]:.2f}")
