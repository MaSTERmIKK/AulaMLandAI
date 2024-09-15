# Ecco un esempio di come definire 
# un modello di regressione utilizzando il dataset Diabetes


#Importare le librerie necessarie
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Caricare il dataset Diabetes
diabetes = load_diabetes()
X = diabetes.data  # Caratteristiche
y = diabetes.target  # Valori target

# Suddividere il dataset in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definire il modello: Regressione Lineare
model = LinearRegression()

# Addestrare il modello sui dati di training
model.fit(X_train, y_train)

# Fare predizioni sui dati di test
y_pred = model.predict(X_test)

# Valutare le prestazioni del modello
mse = mean_squared_error(y_test, y_pred)
print(f"Errore Quadratico Medio del modello: {mse:.2f}")
