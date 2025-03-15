import subprocess
import image_reader 

def solve(grid_digits):
    # sudoku_image_path = "sudoku.jpg"
    input_data = "\n".join(" ".join(map(str, row)) for row in grid_digits)

    result = subprocess.run(
        ["C:\\Users\\mateu\\Documents\\vsc\\soduko.exe"],
        input=input_data, 
        text=True, 
        capture_output=True
    )

    solved_sudoku = result.stdout.strip()
    print("Rozwiązane Sudoku:")
    print(solved_sudoku)
    return solved_sudoku
