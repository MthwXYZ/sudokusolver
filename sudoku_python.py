import subprocess
import image_reader 

sudoku_image_path = "sudokuroz.jpg" 
extracted_sudoku = image_reader.extract_digits(sudoku_image_path)

print("Wykryte cyfry w Sudoku:")
print(extracted_sudoku)

input_data = "\n".join(" ".join(map(str, row)) for row in extracted_sudoku)

result = subprocess.run(
    ["C:\\Users\\mateu\\Documents\\vsc\\soduko.exe"],
    input=input_data, 
    text=True, 
    capture_output=True
)

solved_sudoku = result.stdout.strip()
print("Rozwiązane Sudoku:")
print(solved_sudoku)