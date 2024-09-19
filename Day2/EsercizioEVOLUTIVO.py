import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

# Caricamento del dataset
data = load_wine()
X = data.data
y = data.target

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Creazione della pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('gbc', GradientBoostingClassifier(random_state=42))
])

# Definizione della distribuzione dei parametri
param_dist = {
    'pca__n_components': sp_randint(5, 13),
    'gbc__n_estimators': sp_randint(50, 200),
    'gbc__learning_rate': uniform(0.01, 0.2),
    'gbc__max_depth': sp_randint(1, 5),
    'gbc__subsample': uniform(0.6, 0.4),
    'gbc__min_samples_split': sp_randint(2, 10),
    'gbc__min_samples_leaf': sp_randint(1, 10),
    'gbc__max_features': ['auto', 'sqrt', 'log2', None]
}

# Configurazione della validazione incrociata stratificata
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Configurazione di RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=50, cv=cv,
    scoring='accuracy', random_state=42, n_jobs=-1)

# Avvio della ricerca
random_search.fit(X_train, y_train)

# Migliori parametri trovati
print("Migliori parametri trovati:")
print(random_search.best_params_)

# Predizione sul test set
y_pred = random_search.predict(X_test)

# Report di classificazione
print("Report di classificazione:")
print(classification_report(y_test, y_pred))

# Matrice di confusione
print("Matrice di confusione:")
print(confusion_matrix(y_test, y_pred))

# Accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza sul test set: {accuracy:.4f}")
