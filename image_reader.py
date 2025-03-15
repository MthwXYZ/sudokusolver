import tensorflow as tf
from keras.api.models import load_model
from imutils.perspective import four_point_transform
import numpy as np
import imutils
import cv2
import sudoku_python

# Załaduj model
def load_digit_model():
    return load_model('digit_recognition_model.h5')

# Inicjalizacja kamery
def initialize_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Nie znaleziono kamery")
    return cap

# Przetwarzanie obrazu
def preprocess_frame(frame):
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adTh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 5)
    return adTh, frame

# Znajdowanie linii na obrazie
def find_lines(adTh):
    lines = cv2.HoughLinesP(adTh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines

# Znajdowanie konturu sudoku
def find_sudoku_contour(img_lines, frame):
    cnts = cv2.findContours(img_lines.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.8 <= ar <= 1.2:  # Współczynnik proporcji dla kwadratowego sudoku
                mainCnt = approx
                full_coords = mainCnt.reshape(4, 2)
                return full_coords, mainCnt
    return None

# Zastosowanie transformacji czteropunktowej
def apply_four_point_transform(img_lines, frame, mainCnt):
    sudoku = four_point_transform(img_lines, mainCnt.reshape(4, 2))
    sudoku_clr = four_point_transform(frame, mainCnt.reshape(4, 2))
    sud_c = sudoku.copy()
    return sudoku, sudoku_clr, sud_c

# Podświetlanie siatki sudoku
def highlight_grid(sud_c):
    horizontal = np.copy(sud_c)
    vertical = np.copy(sud_c)

    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    rows = vertical.shape[0]
    verticalsize = rows // 10
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    grid = cv2.bitwise_or(horizontal, vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grid = cv2.dilate(grid, kernel)

    grid = cv2.bitwise_and(grid, sud_c)
    return grid

# Wykrywanie cyfr w oknach
def detect_digits(num, model, rois, smallest_prop_area, buffer_r, buffer_c):
    grid_digits = ['0'] * 81
    i = -1

    windowsize_r = (num.shape[0] // 9) - 1
    windowsize_c = (num.shape[1] // 9) - 1

    for r in range(0, num.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, num.shape[1] - windowsize_c, windowsize_c):
            rois.append([r, r + windowsize_r, c, c + windowsize_c])
            i += 1

            window = num[r + buffer_r: r + buffer_r + windowsize_r, c + buffer_c: c + buffer_c + windowsize_c]

            proposals = cv2.findContours(window, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            proposals = imutils.grab_contours(proposals)

            if len(proposals) > 0:
                digit = sorted(proposals, key=cv2.contourArea, reverse=True)[0]
                perimeter = cv2.arcLength(digit, True)
                approx_shape = cv2.approxPolyDP(digit, 0.02 * perimeter, True)
                bound_rect = cv2.boundingRect(approx_shape)

                rect_area = bound_rect[2] * bound_rect[3]
                if rect_area < smallest_prop_area:
                    continue

                (x, y, w, h) = bound_rect
                s = 2 * (max(w, h) // 2)

                try:
                    prop = cv2.resize(window, (28, 28), cv2.INTER_AREA)
                    prop = np.atleast_3d(prop)
                    prop = np.expand_dims(prop, axis=0)

                    pred = model.predict(prop).argmax(axis=1)
                    grid_digits[i] = str(int(pred[0]) + 1)
                except:
                    pass
    return grid_digits

# Rozwiązywanie sudoku
def solve_sudoku(grid_digits):
    solved = sudoku_python.solve(grid_digits)
    return solved

# Główna funkcja
def main():
    model = load_digit_model()
    cap = initialize_webcam()

    grid_digits = ['0'] * 81
    rois = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        adTh, frame = preprocess_frame(frame)
        lines = find_lines(adTh)
        img_lines = adTh.copy()

        try:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
        except:
            pass

        full_coords, mainCnt = find_sudoku_contour(img_lines, frame)

        if mainCnt is not None:
            sudoku = four_point_transform(img_lines, mainCnt.reshape(4, 2))
            sudoku_clr = four_point_transform(frame, mainCnt.reshape(4, 2))
            grid = highlight_grid(sudoku_clr)

            num = cv2.bitwise_xor(sudoku, grid)
            smallest_prop_area = (num.shape[0] // 9) * (num.shape[1] // 9) // 16
            buffer_r = (num.shape[0] // 9) // 9
            buffer_c = (num.shape[1] // 9) // 9

            grid_digits = detect_digits(num, model, rois, smallest_prop_area, buffer_r, buffer_c)

            if len(grid_digits) == 81:
                solved = solve_sudoku(grid_digits)

                if solved:
                    solved = list(solved.values())

                    for e in range(81):
                        if grid_digits[e] != '0':
                            continue
                        sudoku_clr = cv2.putText(sudoku_clr, solved[e], ((rois[e][2] + rois[e][3]) // 2, rois[e][1]),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

                    h, mask = cv2.findHomography(np.array([[num.shape[1], 0], [0, 0], [0, num.shape[0]], [num.shape[1], num.shape[0]]]),
                                                 full_coords)
                    im_out = cv2.warpPerspective(sudoku_clr, h, (frame.shape[1], frame.shape[0]))
                    final_im = im_out + frame

                    cv2.imshow("sudoku", final_im)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()