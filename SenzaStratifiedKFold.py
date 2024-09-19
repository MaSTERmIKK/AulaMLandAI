from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Caricamento del dataset
data = load_wine()
X = data.data
y = data.target

# Definizione della distribuzione dei parametri
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Configurazione di RandomizedSearchCV senza specificare cv
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Esecuzione della ricerca
random_search.fit(X, y)

# Migliori parametri trovati
print("Migliori parametri (senza StratifiedKFold):")
print(random_search.best_params_)
