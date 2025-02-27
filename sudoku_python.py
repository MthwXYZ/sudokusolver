import subprocess

sudoku = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

input_data = "\n".join(" ".join(map(str, row)) for row in sudoku)

result = subprocess.run(
    ["C:\\Users\\mateu\\Documents\\vsc\\soduko.exe"],  # <-- tu podajemy nazwę twojego programu C
    input=input_data, 
    text=True, 
    capture_output=True
)

# Odczytujemy wynik
solved_sudoku = result.stdout.strip()
print(solved_sudoku)  # Wyświetlamy wynik