from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix

# Caricamento del dataset
data = load_wine()
X = data.data
y = data.target

# Creazione del modello base
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    random_state=42
)

# Validazione incrociata
scores = cross_val_score(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
print("Accuratezza media da cross-validation:", scores.mean())

# Definizione della griglia dei parametri per GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Configurazione di GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Esecuzione della ricerca degli iperparametri
grid_search.fit(X, y)

# Migliori parametri trovati
print("Migliori parametri trovati:", grid_search.best_params_)

# Valutazione del modello con i migliori parametri
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)

# Report di classificazione
print("Report di classificazione:")
print(classification_report(y, y_pred))

# Matrice di confusione
print("Matrice di confusione:")
print(confusion_matrix(y, y_pred))
