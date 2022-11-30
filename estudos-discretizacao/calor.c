#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GLFW/glfw3.h>
// #include "solve-LU.h"

#define WIDTH 600
#define HEIGHT 600

void print_matrix(double** A, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      printf("%lf ", A[i][j]);
    }
    printf("\n");
  }
}

void print_vector(double* A, int n) {
  for(int i = 0; i < n; i++) {
    printf("%lf ", A[i]);
  }
  printf("\n");
}

/*
  Funcao que recebe uma matriz M, um inteiro N e
  desaloca os ponteiros apontados por M, assim como desaloca M.
*/
void desalocaMatriz(double** M, int n) {
  for(int i = 0; i < n; i++) free(M[i]);
  free(M);
}

double** create_matrix(int n, int m) {
  double **mtx;
  mtx = (double**) malloc(n*sizeof(double*));
  for(int i = 0; i < n; i++) {
    mtx[i] = (double*) malloc(m*sizeof(double));
    for(int j = 0; j < m; j++) {
      mtx[i][j] = 0;
    }
  }
  return mtx;
}

double** create_id_matrix(int n) {
  double **mtx = create_matrix(n, n);
  for(int i = 0; i < n; i++) {
    mtx[i][i] = 1;
  }
  return mtx;
}

double** copy_matrix(double** A, int n, int m) {
  double **mtx = create_matrix(n, m);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      mtx[i][j] = A[i][j];
    }
  }
  return mtx;
}

void copy_alloc_matrix(double** A, double** B, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      B[i][j] = A[i][j];
    }
  }
}

/*
  Funcao com o algoritmo de soma de matrizes tradicional.
*/
int soma_matrizes(int n, double** A, double** B, double** C) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return 1;
}

/*
  Funcao com o algoritmo de multiplicacao de matrizes tradicional.
*/
int mult_matrizes(int n, double** A, double** B, double** C) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      C[i][j] = 0;
      for(int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return 1;
}

/*
  Funcao com o algoritmo de multiplicacao de matrizes tradicional.
*/
int mult_matriz_vec(int n, double** A, double* X, double* b) {
  for(int i = 0; i < n; i++) {
    b[i] = 0;
    for(int k = 0; k < n; k++) {
      b[i] += A[i][k] * X[k];
    }
  }
  return 1;
}

// Função que resolve um sistema linear diagonal AX = b
// Recebe uma matriz quadrada A e um vetor b
// Retorna o vetor X, solução do sistema linear diagonal AX = b
void RESOLVE_DIAGONAL(int n, double** A, double* b, double* X) {
  for(int i = 0; i < n; i++) {
    int Xn = n-1-i;
    X[Xn] = b[Xn] / A[Xn][Xn];
  }
}

// Função que resolve um sistema triangular superior AX = b
// Recebe uma matriz quadrada A e um vetor b
// Retorna o vetor X, solução do sistema triangular superior AX = b
void RESOLVE_TRIANGULAR_SUPERIOR(int n, double** A, double* b, double* X) {
  for(int i = 0; i < n; i++) {
    int Xn = n-1-i;

    double sum = 0;
    for(int j = Xn+1; j < n; j++) {
      sum += A[Xn][j]*X[j];
    }

    X[Xn] = ( b[Xn] - sum ) / A[Xn][Xn];
  }
}

// Função que resolve um sistema triangular inferior AX = b
// Recebe uma matriz quadrada A e um vetor b
// Retorna o vetor X, solução do sistema triangular inferior AX = b
void RESOLVE_TRIANGULAR_INFERIOR(int n, double** A, double* b, double* X) {
  for(int i = 0; i < n; i++) {
    double sum = 0;
    for(int j = 0; j < i; j++) {
      sum += A[i][j]*X[j];
    }

    X[i] = ( b[i] - sum ) / A[i][i];
  }
}


// Função que recebe uma posição i,j, monta uma
// matriz E de zeros nxn e substitui E[i,j] por 1
// Retorna a matriz resultante
double** E(int n, int i, int j) {
  double **e = create_matrix(n, n);
  e[i][j] = 1;
  return e;
}

/** Função que implementa a decomposição LU em uma matriz nxn.
    Recebe uma matriz A quadrada nxn
    Retorna L e U da decomposição
*/
void DECOMPOSICAO_LU(int n, double** A, double** rL, double** rU) {
  double **Id = create_id_matrix(n); // Matriz identidade nxn
  double **L = create_id_matrix(n); // Inicializando a matriz L com a matriz identidade
  double **U = copy_matrix(A, n, n); // Inicializando a matriz U com a matriz A

  // Neste laço montamos as matrizes L e U
  // Sendo U o resultado de aplicar a eliminação gaussiana em A
  // e L o inverso da matriz que multiplica A para obter U
  for(int i = 0; i < n-1; i++) {
    for(int j = i+1; j < n; j++) {
      // Acha o fator que multiplica a linha i
      // para fazer o elemento U[j,i] = 0
      double k = -U[j][i] / U[i][i];

      // Expressão matricial, na matriz U, equivalente a multiplicar
      // a linha i por k e somar na linha j
      double **eji = E(n, j, i);
      eji[j][i] = k;

      double **aux1 = create_matrix(n, n);
      double **aux2 = copy_matrix(U, n, n);      

      soma_matrizes(n, Id, eji, aux1);

      desalocaMatriz(U, n);
      U = create_matrix(n, n);
      mult_matrizes(n, aux1, aux2, U);

      // Análogo à expressão acima, porém com a diferença de que
      // ao final L, é a inversa da matriz que multiplica A e obtém U
      eji[j][i] = -k;

      desalocaMatriz(aux1, n);
      desalocaMatriz(aux2, n);

      aux1 = create_matrix(n, n);
      aux2 = copy_matrix(L, n, n);

      soma_matrizes(n, Id, eji, aux1);
      
      desalocaMatriz(L, n);
      L = create_matrix(n, n);
      mult_matrizes(n, aux2, aux1, L);

      // L = L * (Id - k*E(j,i))
    }
  }

  copy_alloc_matrix(U, rU, n, n);
  copy_alloc_matrix(L, rL, n, n);
}

// Função que recebe a matriz A quadrada nxn e o vetor b nx1 do sistema AX = b
// e obtém a solução X utilizando a decomposição LU, retornando-a.
void SOLVE_LU(int n, double** A, double* b, double* X) {
  double **L, **U;
  L = create_matrix(n, n);
  U = create_matrix(n, n);

  double *c = (double*) malloc(n*sizeof(double));

  // Decompõe A = LU
  DECOMPOSICAO_LU(n, A, L, U);

  // Resolve o sistema Lc = b
  RESOLVE_TRIANGULAR_INFERIOR(n, L, b, c);

  // Resolve o sistema UX = c
  RESOLVE_TRIANGULAR_SUPERIOR(n, U, c, X);
}

void preencheMatrizCalorA(int n, double** A, double r) {
  A[0][0] = 2+2*r;
  A[0][1] = -r;
  A[n-1][n-2] = -r;
  A[n-1][n-1] = 2+2*r;

  for(int i = 1; i < n-1; i++) {
    A[i][i-1] = -r;
    A[i][i] = 2+2*r;
    A[i][i+1] = -r;
  }
}

void preencheMatrizCalorB(int n, double** B, double r) {
  B[0][0] = 2-2*r;
  B[0][1] = r;
  B[n-1][n-2] = r;
  B[n-1][n-1] = 2-2*r;

  for(int i = 1; i < n-1; i++) {
    B[i][i-1] = r;
    B[i][i] = 2-2*r;
    B[i][i+1] = r;
  }
}

// Tipo que armazena as coordenadas de um pixel
typedef struct mtx_coords {
    int x;
    int y;
} t_coords;

// Tipo que armazena configurações do plot da fractal na janela
typedef struct window_plot {
    double zoom;
    double XOffset;
    double YOffset;
    int width, height;
} t_plot;

GLubyte PixelBuffer[WIDTH * HEIGHT * 3];

// Matrizes para testar a corretude da forma concorrente do programa
int mtxtestseq[WIDTH * HEIGHT * 3];
int mtxtestconc[WIDTH * HEIGHT * 3];

// Atribuindo configurações do programa
t_plot WindowMatrixPlot = {1, -0, 0, WIDTH, HEIGHT};
int N_THREADS;
int MAX_ITERATIONS = 100;
double MAX_BRIGHT_LENGTH = 1;


// Variável que armazena o andamento do programa na matriz
int mtxposition;

// Recebe o "andamento" da matriz e define a coordenada de onde está
void nextMatrixLocation(int mtxposition, t_coords* fractalMatrixLocation) {
    int x = mtxposition % WIDTH;
    int y = mtxposition / WIDTH;

    fractalMatrixLocation->x = x;
    fractalMatrixLocation->y = y;
}

// Recebe um valor pertencente a um intervalo [min,max] e retorna o valor transformado
// para o intervalo [floor,ceil]
double map(double value, double min, double max, double floor, double ceil) {
    return floor + (ceil - floor) * ((value - min) / (max - min));
}

// "Pinta" os pixels, definidos em PixelBuffer, na janela do programa
void display(GLFWwindow *window) {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);
    glfwSwapBuffers(window);
}

// Atribui ao pixel na posição (x,y) a cor [r,g,b]
void makePixel(int x, int y, int r, int g, int b, GLubyte* pixels, int width, int height) {
    if (0 <= x && x < width && 0 <= y && y < height) {
        int position = (x + y * width) * 3;
        pixels[position] = r;
        pixels[position + 1] = 0;
        pixels[position + 2] = b;
    }
}

void render(int n, double* estado) {
  int barHeight = HEIGHT/3;
  int nodeWidth = WIDTH/n;
  int node = -1;
  // printf("nodeWidth = %d\n", nodeWidth);

  for(int x = 0; x < WIDTH; x++) {
    for(int y = 0; y < HEIGHT; y++) {
      if(y >= barHeight && y < barHeight*2) {
        if(x % nodeWidth == 0 && y == barHeight) node++;

        double bright = map(estado[node], 0, 20, 0, 255);
        // printf("bright = %lf\n", estado[node]);
        makePixel(
          x, y, 
          bright, 0, 255.0-bright,
          PixelBuffer, WIDTH, HEIGHT
        );
      } else {
        makePixel(
          x, y, 
          0, 0, 0,
          PixelBuffer, WIDTH, HEIGHT
        );
      }
    }
  }
}

int main() {
  int n = 30, nt = 200;
  double L, dx, dt, k, r;

  double **A = create_matrix(n, n);
  double **B = create_matrix(n, n);
  double *X = (double*) malloc(n*sizeof(double));
  double *u = (double*) malloc(n*sizeof(double));
  double *b = (double*) malloc(n*sizeof(double));
  double **estados = (double**) malloc(nt*sizeof(double*));
  for(int t = 0; t < nt; t++) {
    estados[t] = (double*) malloc(n*sizeof(double));
  }

  GLFWwindow* window;
  // Inicializando a biblioteca
  if (!glfwInit())
      return -1;

  // Criando a janela e seu contexto OpenGL
  window = glfwCreateWindow(WIDTH, HEIGHT, "Difusão de Calor", NULL, NULL);
  if (!window)
  {
      glfwTerminate();
      return -1;
  }

  // Cria o contexto atual da janela
  glfwMakeContextCurrent(window);

  L = 1.0;
  dx = L/10.0;
  dt = 0.1;
  k = 0.12;
  r = dt*k/(dx*dx);

  for(int i = 0; i < n; i++) {
    u[i] = 20;
    estados[0][i] = 20;
  }

  preencheMatrizCalorA(n, A, r);
  preencheMatrizCalorB(n, B, r);

  mult_matriz_vec(n, B, u, b);
  // solveMatrix(n, A, b, X);
  SOLVE_LU(n, A, b, X);

  // for(int t = 1; t < nt; t++) {
  //   for(int i = 0; i < n; i++) {
  //     u[i] = X[i];
  //     estados[t][i] = X[i];
  //   }
  //   mult_matriz_vec(n, B, u, b);

  //   SOLVE_LU(n, A, b, X);
  // }
  
  // for(int t = 0; t < nt; t++) {
  //   printf("Vector t%d = \n", t);
  //   print_vector(estados[t], n);
  //   printf("\n");
  // }
  int teste;
  int frame = 1;
  while (!glfwWindowShouldClose(window)){
    // Configuração da visualização
    mtxposition = -1;
    glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
    glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);


    for(int i = 0; i < n; i++) {
      u[i] = X[i];
      estados[frame][i] = X[i];
    }
    mult_matriz_vec(n, B, u, b);

    SOLVE_LU(n, A, b, X);

    // Pinta os pixels na janela
    render(n, estados[frame]);

    frame++;
    glClear(GL_COLOR_BUFFER_BIT);

    // Desenhando
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

    // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
    glfwSwapBuffers(window);
    glfwPollEvents();

    if(frame == nt) break;
  }


  return 0;
}