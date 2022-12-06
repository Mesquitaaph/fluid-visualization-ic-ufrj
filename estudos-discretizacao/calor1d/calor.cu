#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GLFW/glfw3.h>

#include "../solve-LU.h"

#define WIDTH 600
#define HEIGHT 600

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

/*
  Funcao com o algoritmo de multiplicacao de matrizes tradicional.
*/
int mult_matriz_vec_local(int n, double** A, double* X, double* b) {
  double *Al = (double*)malloc(n*n*sizeof(double));
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      Al[i*n + j] = A[i][j];
    }
  }
  int status = mult_matriz_vec(n, Al, X, b);

  free(Al);
  return status;
}

int main() {
  int n = 200, nt = 8000;
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

  mult_matriz_vec_local(n, B, u, b);

  // SOLVE_LU(n, A, b, X);
  solveMatrix(n, A, b, X);

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
  char initSim;
  scanf("%c", &initSim);
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
    mult_matriz_vec_local(n, B, u, b);

    // SOLVE_LU(n, A, b, X);
    solveMatrix(n, A, b, X);

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