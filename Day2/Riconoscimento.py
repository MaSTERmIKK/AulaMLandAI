#Certamente! Ecco il codice completo unito per il riconoscimento basilare delle immagini utilizzando il dataset MNIST e una rete neurale convoluzionale (CNN) con TensorFlow e Keras.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Caricamento del dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Pre-elaborazione dei dati
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Reshape per adattarsi ai layer convoluzionali
X_train = X_train.reshape(-1, 28, 28, 1)  # Aggiunta del canale per scala di grigi
X_test = X_test.reshape(-1, 28, 28, 1)

# Conversione delle etichette in formato one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Creazione del modello CNN
model = Sequential()

# Primo layer convoluzionale e di pooling
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Secondo layer convoluzionale e di pooling
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
model.add(Flatten())

# Layer denso
model.add(Dense(128, activation='relu'))

# Layer di output
model.add(Dense(10, activation='softmax'))

# Compilazione del modello
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1)

# Valutazione sul test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Perdita sul test set: {test_loss:.4f}')
print(f'Accuratezza sul test set: {test_accuracy:.4f}')

# Predizione sul test set
predictions = model.predict(X_test)

# Conversione delle predizioni in etichette
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Matrice di confusione
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Visualizzazione della matrice di confusione
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.xlabel('Predizione')
plt.ylabel('Vero Valore')
plt.show()

# Report di classificazione
report = classification_report(true_classes, predicted_classes)
print('Report di Classificazione:')
print(report)

# Visualizzazione di alcune predizioni
num_images = 5
random_indices = np.random.choice(len(X_test), num_images)
plt.figure(figsize=(15,3))
for i, idx in enumerate(random_indices):
    image = X_test[idx].reshape(28, 28)
    true_label = true_classes[idx]
    predicted_label = predicted_classes[idx]
    
    plt.subplot(1, num_images, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'T:{true_label}, P:{predicted_label}')
plt.show()
