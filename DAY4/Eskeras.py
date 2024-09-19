# Importare le librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Caricare il dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualizzare alcune immagini di esempio dal set di addestramento
plt.figure(figsize=(7,7))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Etichetta: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Preprocessare i dati
# Normalizzare i valori dei pixel a un intervallo [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Ridimensionare i dati per adattarli al modello (se necessario)
# Nel caso del modello sequenziale con Flatten, non è necessario reshape

# Convertire le etichette in vettori one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Costruire il modello sequenziale
model = Sequential()
# Appiattire le immagini 28x28 in vettori di 784 elementi
model.add(Flatten(input_shape=(28, 28)))
# Aggiungere uno strato denso con 128 neuroni e attivazione ReLU
model.add(Dense(128, activation='relu'))
# Strato di output con 10 neuroni (uno per classe) e attivazione softmax
model.add(Dense(10, activation='softmax'))

# Compilare il modello
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Stampare un riepilogo del modello
model.summary()

# Addestrare il modello
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1)

# Valutare il modello sui dati di test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nAccuratezza sul set di test:', test_acc)

# Fare predizioni sul set di test
predictions = model.predict(x_test)

# Funzione per visualizzare l'immagine e la predizione
def plot_image_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], np.argmax(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap='gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel(f"Pred: {predicted_label} ({100*np.max(predictions_array):.2f}%)\nTrue: {true_label}", color=color)

# Visualizzare alcune immagini con le loro predizioni
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image_prediction(i, predictions, y_test, x_test)
plt.tight_layout()
plt.show()
