import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_recognition_model.h5")

def preprocess_cell(cell):
    cell = cv2.resize(cell, (28, 28))  # Skalowanie do 28x28
    cell = cell / 255.0  # Normalizacja
    return cell.reshape(1, 28, 28, 1)  # Dopasowanie do CNN

def extract_digits(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Nie udało się załadować obrazu z ścieżki: {image_path}")
    
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("Przetworzony obraz", image)
    
    cells = []
    cell_size = image.shape[0] // 9  # Podział obrazu na 9x9

    for y in range(9):
        row = []
        for x in range(9):
            x_start, y_start = x * cell_size, y * cell_size
            x_end, y_end = (x + 1) * cell_size, (y + 1) * cell_size

            cell = image[y_start:y_end, x_start:x_end]
            processed = preprocess_cell(cell)

            # Przewidujemy cyfrę modelem CNN
            prediction = model.predict(processed)
            digit = np.argmax(prediction)  # Pobieramy najpewniejszą predykcję

            row.append(digit)
        cells.append(row)

    return np.array(cells)