import cv2

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Wczytanie obrazu
image = cv2.imread("sudoku.jpg")

# Konwersja do odcieni szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Rozmycie Gaussa (usuwa szumy)
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Progowanie (thresholding) - zamieniamy obraz na czarno-biały
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Podgląd wyników
cv2.imshow("Oryginał", image)
cv2.imshow("Szarość", gray)
cv2.imshow("Rozmycie", blurred)
cv2.imshow("Progowanie", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()