import tensorflow as tf
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.datasets import mnist
import numpy as np

# 1️⃣ Wczytanie i przygotowanie danych MNIST (zdjęcia cyfr)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2️⃣ Normalizacja danych (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3️⃣ Dodanie wymiaru kanału (28x28 -> 28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Warstwa konwolucyjna
    MaxPooling2D((2,2)),  # Warstwa max pooling
    Flatten(),  # Spłaszczenie danych
    Dense(128, activation='relu'),  # Warstwa w pełni połączona
    Dense(10, activation='softmax')  # Warstwa wyjściowa (10 klas - cyfry 0-9)
])

# 5️⃣ Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6️⃣ Trenowanie modelu
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 7️⃣ Testowanie modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Dokładność na danych testowych: {test_acc:.4f}")

# 8️⃣ Zapis modelu do pliku
model.save("digit_recognition_model.h5")