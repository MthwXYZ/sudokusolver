import tensorflow as tf
from imutils.perspective import four_point_transform
import numpy as np
import imutils
import cv2
import sudoku_python  # Zakładamy, że masz moduł rozwiązujący sudoku

load_model = tf.keras.models.load_model
# Załaduj model rozpoznawania cyfr
def load_digit_model():
    return load_model("digit_recognition_model.h5")

# Wczytaj obraz zamiast kamery
def load_image(image_path="sudoku.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Nie można załadować obrazu: {image_path}")
    return image

# Przetwarzanie obrazu
def preprocess_image(image):
    image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 5)
    return adaptive_thresh, image

# Znajdowanie konturu planszy sudoku
def find_sudoku_contour(thresh_image, original_image):
    contours = cv2.findContours(thresh_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

# Znalezienie cyfr w siatce sudoku
def extract_digits(sudoku_image, model):
    sudoku_gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    cell_size = sudoku_gray.shape[0] // 9  # Zakładamy kwadratowe pole
    digits = []

    for y in range(9):
        for x in range(9):
            x_start, y_start = x * cell_size, y * cell_size
            x_end, y_end = (x + 1) * cell_size, (y + 1) * cell_size
            cell = sudoku_gray[y_start:y_end, x_start:x_end]

            # Przetwarzanie komórki
            cell = cv2.resize(cell, (28, 28))
            cell = cell / 255.0
            cell = cell.reshape(1, 28, 28, 1)

            # Rozpoznawanie cyfry
            prediction = model.predict(cell)
            digit = np.argmax(prediction)

            # Jeśli mała pewność, ustaw jako 0 (puste pole)
            if np.max(prediction) < 0.8:
                digit = 0  

            digits.append(str(digit))

    return digits

# Rozwiązanie sudoku
def solve_sudoku(grid_digits):
    solved = sudoku_python.solve(grid_digits)
    return solved

# Główna funkcja
def main():
    model = load_digit_model()
    image = load_image("sudoku.jpg")

    thresh, processed_image = preprocess_image(image)
    sudoku_contour = find_sudoku_contour(thresh, processed_image)

    if sudoku_contour is not None:
        sudoku_warped = four_point_transform(processed_image, sudoku_contour)
        detected_digits = extract_digits(sudoku_warped, model)

        if len(detected_digits) == 81:
            solved_sudoku = solve_sudoku(detected_digits)
            print("Oryginalna plansza:")
            print(np.array(detected_digits).reshape(9, 9))

            print("\nRozwiązane sudoku:")
            print(np.array(list(solved_sudoku.values())).reshape(9, 9))
        else:
            print("Błąd: Nie udało się wykryć wszystkich cyfr.")
    else:
        print("Nie znaleziono planszy Sudoku.")

if __name__ == "__main__":
    main()