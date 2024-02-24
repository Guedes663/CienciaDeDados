import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

diretorio = r'C:\Users\gabri\OneDrive\Área de Trabalho\Biblioteca\Atividades_e_Trabalhos\atividade_regressao_linear\csv'

nome_arquivo = 'aluguel_uma_variavel_sem_ruido.csv'

caminho_arquivo = os.path.join(diretorio, nome_arquivo)

df = pd.read_csv(caminho_arquivo)

print(df.head())
print(df.describe())

plt.scatter(df['Area'], df['Preco'])
plt.xlabel('Area (m²)')
plt.ylabel('Preco do Aluguel (R$)')
plt.title('Preco do Aluguel X Area')
plt.show()

x = df[['Area']]
y = df[['Preco']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

modelo_regressao_linear = LinearRegression()
modelo_regressao_linear.fit(x_train, y_train)

# Previsões
y_pred = modelo_regressao_linear.predict(x_test)

# Avaliação
print(f"R²: {r2_score(y_test, y_pred):.2f}")

plt.scatter(df['Area'], df['Preco'], color = 'blue')
plt.plot(x_test, y_pred, color='black')
plt.xlabel('Area (m²)')
plt.ylabel('Preco do Aluguel (R$)')
plt.title('Preco do Aluguel x Area')
plt.show()