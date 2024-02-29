from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Carregar o conjunto de dados iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir conjuntos de treino e teste
X_train_val, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

# Instanciando modelo de regressão logística e treinando
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Previsão
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo de regressão logística com normalização: ", accuracy)