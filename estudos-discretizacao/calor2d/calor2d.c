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

  desalocaMatriz(L, n);
  desalocaMatriz(U, n);
  desalocaMatriz(Id, n);
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
  desalocaMatriz(L, n);
  desalocaMatriz(U, n);
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

void render2D(int n, double** estado) {
  int nodeHeight = HEIGHT/n;
  int nodeWidth = WIDTH/n;
  int node_x = -1;
  int node_y = -1;

  for(int x = 0; x < WIDTH; x++) {
    if(x % nodeWidth == 0) node_x++;
    node_y = -1;
    for(int y = 0; y < HEIGHT; y++) {
      if(y % nodeHeight == 0) node_y++;

      double bright = map(estado[node_x][node_y], 0, 20, 0, 255);
      makePixel(
        x, y, 
        bright, 0, 255.0-bright,
        PixelBuffer, WIDTH, HEIGHT
      );
    }
  }
}

int e_N(int i) {
  return 0 ? i == 0 : 1;
}

int e_S(int i, int n) {
  return 0 ? i == n-2 : 1;
}

int e_W(int j) {
  return 0 ? j == 0 : 1;
}

int e_E(int j, int n) {
  return 0 ? j == n-2 : 1;
}

double nova_temp_aux(int n, double** T, int i, int j, int r) {
  if(i == 0 || i == n-1 || j == 0 || j == n-1) return 0.0;
  double cond_terms = e_S(i, n)*T[i+1][j] + e_N(i)*T[i-1][j] + e_E(j, n)*T[i][j+1] + e_W(j)*T[i][j-1];
  return ((2-4*r)*T[i][j] + r*cond_terms)/2;
}

void nova_temp(int n, double** T, double** Tn, int i, int j, int r) {
  double cond_terms_n = e_S(i, n)*T[i+1][j] + e_N(i)*T[i-1][j] + e_E(j, n)*T[i][j+1] + e_W(j)*T[i][j-1];
  double cond_terms_n1 = 
    nova_temp_aux(n, T, i+1, j, r) + 
    nova_temp_aux(n, T, i-1, j, r) + 
    nova_temp_aux(n, T, i, j+1, r) + 
    nova_temp_aux(n, T, i, j-1, r);
  Tn[i][j] = ((2-4*r)*T[i][j] + r*(cond_terms_n + cond_terms_n1))/(2+4*r);
}

double sol_analitica(double x, double y, double t, double k) {
  double pi = 3.141592653589793;
  double max = 50.0;

  double sum = 0;
  for(double m = 1; m < max; m++) {
    double mu = (m*pi);

    for(double n = 1; n < max; n++) {
      double Amn = (80.0*(pow(-1, n-1)+1)*(pow(-1, m-1) + 1))/(pi*pi*m*n);
      double v = (n*pi);
      // printf("v = %lf\n", v);
      double lambda = sqrt(k)*sqrt(mu*mu + v*v);

      sum += Amn*sin(mu*x)*sin(v*y)*exp(-1.0*lambda*lambda*t);
    }
  }

  return sum;
}

int main() {
  int n = 100, nt = 200;
  double L, dx, dt, k, r;

  double **T = create_matrix(n, n);
  double **Tn = create_matrix(n, n);

  // GLFWwindow* window;
  // // Inicializando a biblioteca
  // if (!glfwInit())
  //   return -1;

  // // Criando a janela e seu contexto OpenGL
  // window = glfwCreateWindow(WIDTH, HEIGHT, "Difusão de Calor 2d", NULL, NULL);
  // if (!window)
  // {
  //     glfwTerminate();
  //     return -1;
  // }

  // // // Cria o contexto atual da janela
  // glfwMakeContextCurrent(window);

  L = 1.0;
  dx = L/10.0;
  dt = 0.1;
  k = 0.12;
  r = dt*k/(dx*dx);

  for(int i = 1; i < n-1; i++) {
    for(int j = 1; j < n-1; j++) {
      T[i][j] = 20.0;
    }
  }

  double erro_aprox = 20.0 - sol_analitica(0.5, 0.5, 0, k);
  for(int t = 0; t < 100; t++) {
    for(int i = 1; i < n-1; i++) {
      for(int j = 1; j < n-1; j++) {
        nova_temp(n, T, Tn, i, j, r);
      }
    }
    printf("SOL_APX = %lf\n", T[n/2][n/2]);
    printf("SOL_ANL = %lf\n\n", sol_analitica(0.5, 0.5, t*0.1, k));

    copy_alloc_matrix(Tn, T, n, n);
  }
  // char initSim;
  // scanf("%c", &initSim);
  int frame = 1;
  // while (!glfwWindowShouldClose(window)){
  //   // Configuração da visualização
  //   glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
  //   glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);
  //   for(int i = 1; i < n-1; i++) {
  //     for(int j = 1; j < n-1; j++) {
  //       nova_temp(n, T, Tn, i, j, r);
  //     }
  //   }


  //   // Pinta os pixels na janela
  //   render2D(n, T);
  //   copy_alloc_matrix(Tn, T, n, n);

  //   frame++;
  //   glClear(GL_COLOR_BUFFER_BIT);

  //   // Desenhando
  //   glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

  //   // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
  //   glfwSwapBuffers(window);
  //   glfwPollEvents();

  //   if(frame == nt) break;
  // }

  // // for(int i = 0; i <= n; i++) {
  // //   for(int j = 0; j <= n; j++) {
  // //     printf("%lf ", T[i][j] - sol_analitica(i/n, j/n, frame*dt, k));
  // //   }
  // //   printf("\n");
  // // }

  // printf("aprox = %lf, sol = %lf\n", T[49][49], sol_analitica(0.5, 0.5, frame*dt, k*k));

  return 0;
}