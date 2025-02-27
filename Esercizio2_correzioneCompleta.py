# Importazione delle librerie necessarie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# 1. Caricamento del dataset Wine
wine = load_wine()
X = wine.data
y = wine.target
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# 2. Esplorazione del dataset
print("Forma del dataset:", df.shape)
print("\nNumero di campioni per classe:")
print(df['target'].value_counts())
print("\nStatistiche di base delle feature:")
print(df.describe())

# Visualizzazione: Grafico a barre per la distribuzione delle classi
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribuzione delle classi nel dataset Wine')
plt.xlabel('Classe')
plt.ylabel('Numero di campioni')
plt.show()

# 3. Riduzione della dimensionalità con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizzazione: Grafico scatter 2D delle due componenti principali
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=100)
plt.title('PCA: 2 Componenti Principali')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Classe')
plt.show()

# 4. Suddivisione del dataset in training e test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Applicazione di un algoritmo di classificazione: RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 6. Valutazione della performance del modello
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nValutazione del modello Random Forest:")
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualizzazione dell'importanza delle feature
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=wine.feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Importanza delle feature secondo Random Forest")
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.show()

# 8. Visualizzazione della matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Matrice di Confusione")
plt.xlabel("Classe Predetta")
plt.ylabel("Classe Reale")
plt.show()

# 9. Ottimizzazione del modello con GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Migliori parametri trovati:", grid_search.best_params_)
print("Miglior accuratezza ottenuta (CV): {:.4f}".format(grid_search.best_score_))
