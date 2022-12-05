#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#define WIDTH 600
#define HEIGHT 600

// Tipo que armazena configurações do plot da fractal na janela
typedef struct window_plot {
    double zoom;
    double XOffset;
    double YOffset;
    int width, height;
} t_plot;

GLubyte PixelBuffer[WIDTH * HEIGHT * 3];

// Atribuindo configurações do programa
t_plot WindowMatrixPlot = {1, -0, 0, WIDTH, HEIGHT};
int N_THREADS;
int MAX_ITERATIONS = 100;
double MAX_BRIGHT_LENGTH = 1;

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

int main(int argc, char *argv[]) {

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

  int frame = 0;
  while (!glfwWindowShouldClose(window)){
    // Configuração da visualização
    glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
    glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);


    // for(int i = 0; i < n; i++) {
    //   u[i] = X[i];
    //   estados[frame][i] = X[i];
    // }
    // mult_matriz_vec(n, B, u, b);

    // SOLVE_LU(n, A, b, X);

    // Pinta os pixels na janela
    // render(n, estados[frame]);
    makePixel(
      frame, frame, 
      frame%256, 0, 255-(frame%256),
      PixelBuffer, WIDTH, HEIGHT
    );

    frame++;
    glClear(GL_COLOR_BUFFER_BIT);

    // Desenhando
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

    // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
    glfwSwapBuffers(window);
    glfwPollEvents();

    if(frame == 600) break;
  }


  return 1;
}