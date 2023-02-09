#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GLFW/glfw3.h>

#define WIDTH 260
#define HEIGHT WIDTH
#define N_PARTICULAS 1024

typedef struct Particula {
  double x, y, vx, vy;
} t_particula;

void read_file(char * file_name, t_particula **velocidades){
	FILE *fp;
	char ch[50];
	char* s, *e;
  char* aux;
  char var;
	int k, eqk, i = 0, j = -1, vel_param = -1;
  double vel;
	fp = fopen(file_name, "r"); // read mode

	if (fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}

	fscanf(fp, "%s", ch);
	while(!feof(fp)){ 
    eqk = 0;

		s = strtok(ch, "\n");
    // printf("s = %s\n", s);
    for(k = 0;; k++) {
      if(s[k] == '=') {
        eqk = k+1;
      }
      if(s[k] == '\0') break;
    }
    var = s[1];

    aux = (char*) malloc((k-eqk+1)*sizeof(char));

    for(k = eqk;; k++) {
      if(s[k] == '\0') break;
      aux[k-eqk] = s[k];
    }
    
    vel = strtod(aux, &e);

    vel_param = (vel_param + 1)%2;
    if(vel_param == 0) j++;

    if(i == 257) {
      break;
    }

    if(j == 0) {
      continue;
    }

    if(j == 257) {
      if(vel_param == 1) {
        j = -1;
        i++;
      }
      continue;
    }

    if(i == 0) {
      continue;
    }
    
    switch(vel_param){
      case 0:
        velocidades[i-1][j-1].vx = vel;
        printf("velocidades[%d][%d].vx = %.10lf (vel_param = %d)\n", i-1, j-1, vel, vel_param);
        break;
      case 1:
        velocidades[i-1][j-1].vy = vel;
        printf("velocidades[%d][%d].vy = %.10lf (vel_param = %d)\n", i-1, j-1, vel, vel_param);
        break;
      default:
        printf("Hm\n");
    }

		fscanf(fp, "%s", ch);
    if(vel_param == 0 && var != 'x' || vel_param == 1 && var != 'y') {
      printf("Erro\n");
      printf("%s\n", s);
      printf("vel_param = %d\n", vel_param);
      exit(1);
    }
	}

	fclose(fp);
}

void read_file_2(char * file_name, t_particula **velocidades){
	FILE *fp;
	char ch[50], i[4], j[4];
	char* s, *e;
  char* aux;
  char var;
	int k, eqk, aux_ind, int_i, int_j;
  double vel;
	fp = fopen(file_name, "r"); // read mode

	if (fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}

	fscanf(fp, "%s", ch);
	while(!feof(fp)){ 
    eqk = 0;

		s = strtok(ch, "\n");
    
    var = s[1];
    for(k = 0;; k++) {
      if(s[k] == '=') {
        eqk = k+1;
      }
      if(s[k] == '\0') break;
    }
    aux = (char*) malloc((k-eqk+1)*sizeof(char));

    // Descobre o índice i
    aux_ind = 0;
    for(k = 3;; k++) {
      if(s[k] == ']') {
        k += 2;
        break;
      }

      i[aux_ind++] = s[k];
    }
    i[aux_ind] = '\0';

    // Descobre o índice j
    aux_ind = 0;
    for(;; k++) {
      if(s[k] == ']') break;

      j[aux_ind++] = s[k];
    }
    j[aux_ind] = '\0';

    for(k = eqk;; k++) {
      if(s[k] == '\0') break;
      aux[k-eqk] = s[k];
    }
    
    vel = strtod(aux, &e);
    
    int_i = atoi(i), int_j = atoi(j);

    
    if(var == 'x') velocidades[int_i+1][int_j+1].vx = vel;
    if(var == 'y') velocidades[int_i+1][int_j+1].vy = vel;
    
    // if(int_i > 0 && int_i < WIDTH+1 && int_j > 0 && int_j < HEIGHT+1) {
    //   if(var == 'x') velocidades[int_i-1][int_j-1].vx = vel;
    //   if(var == 'y') velocidades[int_i-1][int_j-1].vy = vel;
    // }

		fscanf(fp, "%s", ch);
	}

	fclose(fp);
}

/*
  Funcao que recebe uma matriz M, um inteiro N e
  desaloca os ponteiros apontados por M, assim como desaloca M.
*/
void desalocaMatriz(double** M, int n) {
  for(int i = 0; i < n; i++) free(M[i]);
  free(M);
}

t_particula** create_matrix(int n, int m) {
  t_particula **mtx;
  mtx = (t_particula**) malloc(n*sizeof(t_particula*));
  for(int i = 0; i < n; i++) {
    mtx[i] = (t_particula*) malloc(m*sizeof(t_particula));
    for(int j = 0; j < m; j++) {
      mtx[i][j].x = 0, mtx[i][j].y = 0, mtx[i][j].vx = 0, mtx[i][j].vy = 0;
    }
  }
  return mtx;
}

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

// Recebe um valor pertencente a um intervalo [min,max] e retorna o valor transformado
// para o intervalo [floor,ceil]
double map(double value, double min, double max, double floor, double ceil) {
    return floor + (ceil - floor) * ((value - min) / (max - min));
}

// Atribui ao pixel na posição (x,y) a cor [r,g,b]
void makePixel(int x, int y, int r, int g, int b, GLubyte* pixels, int width, int height) {
  if (0 <= x && x < width && 0 <= y && y < height) {
    int position = (x + y * width) * 3;
    pixels[position] = r;
    pixels[position + 1] = g;
    pixels[position + 2] = b;
  }
}

void render(t_particula** velocidades, t_particula *particulas) {
  int tam_p = 0;

  for(int z = 0; z < N_PARTICULAS; z++) {
    t_particula p = particulas[z];
    int x = p.x;
    int y = p.y;

    for(int i = -tam_p; i <= tam_p; i++) {
      for(int j = -tam_p; j <= tam_p; j++) {
        makePixel(
          x+i, y+j, 
          255, 255, 255,
          PixelBuffer, WIDTH, HEIGHT
        );
      }
    }
  }
}

void preencheMatrizVelA(int n, t_particula** A) {
  int i, j = 0;
  double svx, velx, svy, vely;
  double mag = 10.0;

  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++) {
      A[i][j].vx = -0.0;
      A[i][j].vy = 0.1;

      // svx = pow(-1, rand() % 2);
      // velx = i == 0 ? 1.1 : velx + (double)(rand() % 500)/mag * svx;

      // if(i+velx < 1 || i+velx > n-1) velx *= -1;

      // A[i][j].vx = velx;

      // svy = pow(-1, rand() % 2);
      // vely = j == 0 ? 1.1 : vely + (double)(rand() % 500)/mag;// * svy;
      
      // if(j+vely < 1 || j+vely > n-1) vely *= -1;

      // A[i][j].vy = vely;
    }
  }
}

void atualiza_particulas(double dt, int n_dim, t_particula** A, t_particula* particulas) {
  int tam_p = 0;
  for(int i = 0; i < N_PARTICULAS; i++) {
    t_particula p = particulas[i];

    int x = p.x;
    int y = p.y;

    // Limpa o espaco em que a particula estava
    for(int i = -tam_p; i <= tam_p; i++) {
      for(int j = -tam_p; j <= tam_p; j++) {
        makePixel(
          x+i, y+j, 
          0, 0, 0,
          PixelBuffer, WIDTH, HEIGHT
        );
      }
    }

    t_particula pvel = A[(int)p.x][(int)p.y];

    particulas[i].vx = pvel.vx;
    particulas[i].vy = pvel.vy;

    double aux = particulas[i].x;
    particulas[i].x += pvel.vx * dt;

    if(particulas[i].x < 1) particulas[i].x = 1;
    if(particulas[i].x > n_dim-1) particulas[i].x = n_dim-1;

    particulas[i].y += pvel.vy * dt;
    if(particulas[i].y < 1) particulas[i].y = 1;
    if(particulas[i].y > n_dim-1) particulas[i].y = n_dim-1;
  }
}

int main(int argc, char** argv) {
  int n_dim = WIDTH, nt = 20000;
  double dt = 0.8;//0.0019073486;
  t_particula **A = create_matrix(n_dim, n_dim);

  t_particula particulas[N_PARTICULAS];
  for(int i = 0; i < N_PARTICULAS; i++) {
    int sq = (int)sqrt(N_PARTICULAS);

    // particulas[i].x = WIDTH*0.9;
    // particulas[i].y = HEIGHT*0.1;
    particulas[i].x = (i % sq) * WIDTH/sq;
    particulas[i].y = (i / sq) * HEIGHT/sq;
    particulas[i].vx = 0;
    particulas[i].vy = 0;
  }

  GLFWwindow* window;
  // Inicializando a biblioteca
  if (!glfwInit())
    return -1;

  // Criando a janela e seu contexto OpenGL
  window = glfwCreateWindow(WIDTH, HEIGHT, "Estudos de visualizacao de movimento de particulas", NULL, NULL);
  if (!window){
    glfwTerminate();
    return -1;
  }

  // Cria o contexto atual da janela
  glfwMakeContextCurrent(window);

  read_file_2(argv[1], A);
  // preencheMatrizVelA(n_dim, A);
  
  // int initSim;
  // scanf("%c", &initSim);
  int frame = 1;
  while (!glfwWindowShouldClose(window)){
    // Configuração da visualização
    glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
    glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);
    // printf("x = %lf, y = %lf\n", particulas[0].x, particulas[0].y);
    // Atualiza a posicao de cada uma das particulas
    atualiza_particulas(dt, n_dim, A, particulas);

    // // Pinta os pixels na janela
    render(A, particulas);


    // preencheMatrizVelA(n_dim, A);

    frame++;
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    // Desenhando
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

    // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
    glfwSwapBuffers(window);
    glfwPollEvents();

    if(frame == nt) break;
  }
  printf("Fim\n");
  int fimSim;
  scanf("%c", &fimSim);
  return 0;
}