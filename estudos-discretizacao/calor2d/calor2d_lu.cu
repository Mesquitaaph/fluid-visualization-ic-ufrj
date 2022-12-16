#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GLFW/glfw3.h>

#include "../solve-LU.h"

#define WIDTH 600
#define HEIGHT 600

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
      // printf("node_x = %d, node_y = %d\n", node_x, node_y);
      makePixel(
        x, y, 
        bright, 0, 255.0-bright,
        PixelBuffer, WIDTH, HEIGHT
      );
    }
    // printf("TESTE x = %d, node_x = %d\n", x, node_x);
  }

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

void copy_alloc_matrix_local(double** A, double** B, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      B[i][j] = A[i][j];
    }
  }
}

double sol_analitica(double x, double y, double t, double k) {
  double pi = 3.14159265359;
  int max = 2000;

  double sum = 0;
  for(int m = 1; m < max; m++) {
    double mu = (m*pi)/1;

    for(int n = 1; n < max; n++) {
      double Amn = (80*(pow(-1, n-1)+1)*(pow(-1, m-1) + 1))/(pi*pi*m*n);
      double v = (n*pi)/1;
      double lambda = sqrt(k)*sqrt(mu*mu + v*v);

      sum += Amn*sin(mu*x)*sin(v*y)*exp(-lambda*lambda*t);
    }
  }

  return sum;
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]), nt = 200;
  double L, dx, dt, k, r;

  double **A = create_matrix(n, n);
  double **B = create_matrix(n, n);
  double *X = (double*) malloc(n*sizeof(double));
  double *b = (double*) malloc(n*sizeof(double));
  double *aux = (double*) malloc(n*sizeof(double));
  double **T = create_matrix(n, n);
  double **T_intermed = create_matrix(n, n);
  double **Tn = create_matrix(n, n);

  GLFWwindow* window;
  // Inicializando a biblioteca
  if (!glfwInit())
    return -1;

  // Criando a janela e seu contexto OpenGL
  window = glfwCreateWindow(WIDTH, HEIGHT, "Difusão de Calor 2d", NULL, NULL);
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
    for(int j = 0; j < n; j++) {
      // if(i == 0 || j == 0 || i == n-1 || j == n-1) {
      //   T[i*n + j] = 0.0;
      //   continue;
      // }
      T[i][j] = 20.0;
    }
  }

  preencheMatrizCalorA(n, A, r);
  preencheMatrizCalorB(n, B, r);

  
  // for(int j = 0; j < n; j++) {
  //   for(int i = 0; i < n; i++) {
  //     aux[i] = T[i][j];
  //   }
  //   mult_matriz_vec(n, B, aux, b);
  //   SOLVE_LU(n, A, b, X);
  //   for(int i = 0; i < n; i++) {
  //     T_intermed[i][j] = X[i];
  //   }
  // }

  // for(int i = 0; i < n; i++) {
  //   for(int j = 0; j < n; j++) {
  //     aux[j] = T_intermed[i][j];
  //   }
  //   mult_matriz_vec(n, B, aux, b);
  //   SOLVE_LU(n, A, b, X);
  //   for(int j = 0; j < n; j++) {
  //     Tn[i][j] = X[j];
  //   }
  // }

  // copy_alloc_matrix(Tn, T, n, n);

  // print_matrix(Tn, n, n);

  // Verifica corretude da aproximação
  // for(double i = 0; i < n; i++) {
  //   for(double j = 0; j < n; j++) {
  //     double tam = (double)n+1.0;  
  //     double erro_aprox = sol_analitica((i+1.0)/tam, (j+1.0)/tam, 0.1, k);
  //     printf("%.4lf ", erro_aprox);
  //   }
  //   printf("\n");
  // }

  // solveMatrix(n, A, b, X);
  // print_vector(b, n*n);

  // char initSim;
  // scanf("%c", &initSim);
  int frame = 1;
  while (!glfwWindowShouldClose(window)){
    // Configuração da visualização
    glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
    glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);
    for(int j = 0; j < n; j++) {
      for(int i = 0; i < n; i++) {
        aux[i] = T[i][j];
      }
      mult_matriz_vec_local(n, B, aux, b);
      solveMatrix(n, A, b, X);
      
      for(int i = 0; i < n; i++) {
        T_intermed[i][j] = X[i];
      }
    }

    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        aux[j] = T_intermed[i][j];
      }
      mult_matriz_vec_local(n, B, aux, b);
      solveMatrix(n, A, b, X);
      for(int j = 0; j < n; j++) {
        Tn[i][j] = X[j];
      }
    }


    // Pinta os pixels na janela
    render2D(n, T);
    // printf("TESTE\n");
    copy_alloc_matrix_local(Tn, T, n, n);

    frame++;
    glClear(GL_COLOR_BUFFER_BIT);

    // Desenhando
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

    // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
    glfwSwapBuffers(window);
    glfwPollEvents();

    if(frame == nt) break;
  }

  // for(int i = 0; i <= n; i++) {
  //   for(int j = 0; j <= n; j++) {
  //     printf("%lf ", T[i][j] - sol_analitica(i/n, j/n, frame*dt, k));
  //   }
  //   printf("\n");
  // }

  // printf("aprox = %lf, sol = %lf\n", T[49][49], sol_analitica(0.5, 0.5, frame*dt, k*k));

  return 0;
}