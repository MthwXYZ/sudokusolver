#include <stdio.h>

typedef struct {
    int row;
    int col;
} Cell;

void push(Cell* stack, int *top, Cell cell){
stack[++(*top)] = cell;
}

Cell pop(Cell* stack, int *top){
    return stack[(*top)--];
}

int isEmpty(int top){
    return top == -1;
}


void writeout(int board[9][9]){   //print the board
    int i,j;
    for (i=0; i<9; i++){
        for(j=0; j<9; j++){
            printf("   %d", board[i][j]);
        }
        printf(" \n");
    }
}

int findzero(int board[9][9], int* a, int* b){   // finding empty(0) space
    int i, j;
    for (i= 0; i<9; i++){
        for(j=0; j<9; j++){
            if (board[i][j] == 0){
                *a = i;
                *b = j;
                return 1;
            }
        }
    }
    return 0;
}

int correction(int board[9][9], int a, int b, int num)  //check if number can be correctly placed
{ 
    int i,j;
    int startrow = (a/3) * 3;
    int startcol = (b/3) * 3;    

    for(j=0; j<9; j++){     //checking column and row
        if ((board[a][j] == num) || (board[j][b] == num) )
        return 0;
    }

    /*for(i=0; i<9; i++){     //checking row
        if (board[i][b] == num)
        return 0;
    }
*/
    for (i= 0; i<3; i++){    //checking 3x3
        for(j=0; j<3; j++){
            if (board[startrow + i][startcol + j] == num)
            return 0;
    
        }
    }
   // printf(" %d", board[a][b]);
    return 1;
}

void solve(int board[9][9],int* a, int* b){

    Cell current;
    Cell stack[81];
    int top = -1;


    int num = 1;
    while(findzero(board, a, b) == 1){
    int i = *a;   //to do poprawy bo kloc jest bezsens ale podejrzewam ze adres tego a tu sie wjebie czy coś w tym stylu &a i bedzie git
    int j = *b;   // tu to samo ale to do poprawy jak backtracking ogarne ok
    while (num < 10 && board[i][j] == 0){    // tu te logike sprawdzic wsm 
        
        if (correction(board, i, j, num) == 1){
            board[i][j] = num;
            current.row=j;
            current.col=i;
            push(stack, &top, current);  //tu ogarnac 
            num = 1;
            //printf("%d", board[i][j]);
            }
            else{
                num++;
        }
    }
    if (num > 9){                           // osobno przechować info w stosie o komorkach ktore byly wypelnione wczesniej

        current = pop(stack, &top);
        i = current.col;
        j = current.row;
        num = board[i][j] + 1;
        board[i][j] = 0;
        // tu ma sciagac ze stosu góre i podnosic to dalej
    }
    }
writeout(board);
}
/*co do zrobienia:
  opytymalizacja, ogar stosu, pointery wycieki itp ogarnac
  linijka po linijce przeleciec i zobaczyc co jest 5  

*/

int main(){

int a = 0;
int b = 0;
int sudoku[9][9];

for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
        scanf("%d", &sudoku[i][j]);
    }
}

solve(sudoku, &a, &b);


}