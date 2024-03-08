import pandas as pd
import matplotlib.pyplot as plt

#Carregar o conjunto de dados
url_do_arquivo = "C:\\Users\\1-22-10794\\Desktop\\atividade\\atividade_kmeans\\csv\\BankChurners.csv"

dados = pd.read_csv(url_do_arquivo)
#print(dados.describe())
#print(dados.info())
#print(dados.head())

#Verifica valores ausentes
valores_ausentes = dados.isna().sum()

print(valores_ausentes)

#Removendo colunas irrelevantes
dados = dados['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'].drop()

# Visualizar boxplots para identificar outliers
plt.figure(figsize=(10, 6))
dados.boxplot()
plt.xticks(rotation=90)
plt.title("Boxplots das colunas numéricas")
plt.show()

#CLIENTNUM é o id
#Attrition_Flag se o cliente é usuário do banco
#Customer_Age é a variável demográfica - Idade do Cliente em Anos
#Gender genêro
#Dependent_count números de dependentes
#Education_Level nivel de educação
#Marital_Status se é casado, solteiro, divorciado e não respondido 
#Income_Category renda anual do título da conta
#Card_Category tipo de cartão
#Months_on_book Período de relacionamento com o banco