#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GLFW/glfw3.h>

#define OFFSET_X 1
#define OFFSET_Y 1

#define WIDTH (512+2*OFFSET_X+2*OFFSET_Y)
#define HEIGHT WIDTH

#define MIRROR_WIDTH WIDTH
#define MIRROR_HEIGHT 0

#define TAM_RASTRO 200

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
// #define N_PARTICULAS 128*128

typedef struct Particula {
  double x, y, vx, vy;
  double *rastro_x, *rastro_y;
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
  double vel, qt_vel = 0, med_vel = 0, max_vel = 0, min_vel = 2;
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

    
    // if(var == 'x') velocidades[514-int_j][int_i+1].vx = vel; // 1 -> 514
    // if(var == 'y') velocidades[514-int_j][int_i+1].vy = vel;
    if(var == 'x') velocidades[int_i+OFFSET_X][int_j+OFFSET_Y].vx = vel;
    if(var == 'y') velocidades[int_i+OFFSET_X][int_j+OFFSET_Y].vy = vel;
    
    // if(int_i > 0 && int_i < WIDTH+1 && int_j > 0 && int_j < HEIGHT+1) {
    //   if(var == 'x') velocidades[int_i-1][int_j-1].vx = vel;
    //   if(var == 'y') velocidades[int_i-1][int_j-1].vy = vel;
    // }

    if(var == 'y') {
      double mag_vel = sqrt(velocidades[int_i+1][int_j+1].vx*velocidades[int_i+1][int_j+1].vx + velocidades[int_i+1][int_j+1].vy*velocidades[int_i+1][int_j+1].vy);

      if(mag_vel > max_vel && mag_vel < 2.0) max_vel = mag_vel;
      if(mag_vel < min_vel && mag_vel > 0.000001) min_vel = mag_vel;

      qt_vel++;
      med_vel += mag_vel;
    }

		fscanf(fp, "%s", ch);
	}

	fclose(fp);

  med_vel = med_vel/qt_vel;
  printf("max_vel = %lf, min_vel = %lf, med_vel = %lf\n", max_vel, min_vel, med_vel);
}

double mirror_value(double value, double mirror) {
  if(mirror == 0.0) return value;
  return mirror - value;
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
  if(value < min) return floor;
  if(value > max) return ceil;
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

// Atribui ao pixel na posição (x,y) a cor [r,g,b]
void fadeOutPixel(int x, int y, int magnitude, GLubyte* pixels, int width, int height) {
  int min = 10;
  if (0 <= x && x < width && 0 <= y && y < height) {
    int position = (x + y * width) * 3;
    pixels[position] = pixels[position] == 0 ? 0 : MAX(pixels[position] - magnitude, min);
    pixels[position + 1] = pixels[position + 1] == 0 ? 0 : MAX(pixels[position + 1] - magnitude, min);
    pixels[position + 2] = pixels[position + 2] == 0 ? 0 : MAX(pixels[position + 2] - magnitude, min);

    // printf("(%d, %d, %d)\n", pixels[position], pixels[position + 1], pixels[position + 2]);
  }
}

void render_pixel(int tam_p, double x, double y, double r, double g, double b) {
  for(int i = -tam_p; i <= tam_p; i++) {
    for(int j = -tam_p; j <= tam_p; j++) {
      makePixel(
        x+i, y+j, 
        r, g, b,
        PixelBuffer, WIDTH, HEIGHT
      );
    }
  }
}

void render(t_particula** velocidades, t_particula *particulas, long int N_PARTICULAS) {
  int tam_p = 0;

  for(long int z = 0; z < N_PARTICULAS; z++) {
    t_particula p = particulas[z];
    // int x = p.x;
    // int y = HEIGHT - p.y;

    // double vel = sqrt(p.vx*p.vx + p.vy*p.vy);
    // double bright_r = 255; //map(vel, 0.139557, 2, 0, 255);
    // double bright_g = 255; //map(vel, 0.0, 0.139557, 0, 255);
    // double bright_b = 255;

    for(int r = 0; r < TAM_RASTRO; r++) {
      int x = p.rastro_x[r];
      int y = mirror_value(p.rastro_y[r], MIRROR_HEIGHT);

      // printf("(%d, %d) ", x, y);
      // if(r == TAM_RASTRO - 1) {
      //   render_pixel(tam_p, x, y, 0, 0, 0);
      //   continue;
      // }

      double bright = map(r, 0, TAM_RASTRO-1, 255, 0);

      // printf("%lf ", bright);
      double bright_r = bright;
      double bright_g = bright;
      double bright_b = bright;

      render_pixel(tam_p, x, y, bright_r, bright_g, bright_b);
    }
    // printf("\n");
  }
}

void atualiza_particulas(double dt, int n_dim, t_particula** A, t_particula* particulas, long int N_PARTICULAS) {
  int tam_p = 0;

  for(long int i = 0; i < N_PARTICULAS; i++) {
    t_particula p = particulas[i];

    int x = p.x;
    int y = mirror_value(p.y, MIRROR_HEIGHT);

    t_particula pvel = A[(int)mirror_value(p.y, MIRROR_WIDTH)][(int)p.x];

    particulas[i].vx = pvel.vx;
    particulas[i].vy = pvel.vy;

    particulas[i].x += pvel.vx * dt;

    if(particulas[i].x < 1) particulas[i].x = 1;
    if(particulas[i].x > n_dim-1) particulas[i].x = n_dim-1;

    particulas[i].y += pvel.vy * dt;
    if(particulas[i].y < 1) particulas[i].y = 1;
    if(particulas[i].y > n_dim-1) particulas[i].y = n_dim-1;
    
    int novo_x = particulas[i].x;
    int novo_y = mirror_value(particulas[i].y, MIRROR_HEIGHT);
    if(x != novo_x || y != novo_y) {
      // printf("(%d, %d) -> (%d, %d)\n",x, y, novo_x, novo_y);
      // Atualiza a posição de cada pixel do rastro
      for(int r = TAM_RASTRO - 1; r > 0; r--) {
        // Teoricamente, apaga o ultimo pixel do rastro
        // if(r == TAM_RASTRO - 1) render_pixel(tam_p, p.rastro_x[r], p.rastro_y[r], 0, 0, 0);
        p.rastro_x[r] = p.rastro_x[r-1];
        p.rastro_y[r] = p.rastro_y[r-1];
      }

      p.rastro_x[0] = novo_x;
      p.rastro_y[0] = novo_y;

      render(A, particulas, N_PARTICULAS);
    }
  }
}

void render_2(t_particula** velocidades, t_particula *particulas, long int N_PARTICULAS) {
  int tam_p = 0, fadeOutMagnitude = 0.9;

  for(int x = 0; x < WIDTH; x++) {
    for(int y = 0; y < HEIGHT; y++) {
      fadeOutPixel(
        x, y, 
        fadeOutMagnitude,
        PixelBuffer, WIDTH, HEIGHT
      );
    }
  }

  for(long int z = 0; z < N_PARTICULAS; z++) {
    t_particula p = particulas[z];
    int x = p.x;
    int y = mirror_value(p.y, MIRROR_HEIGHT);
    double color_up_lim = 256;
    int color_trans_mag = 64; // color_transition_magnitude
    double color_down_lim = 256 - (256-color_up_lim) - color_trans_mag;

    double full_color = 256*3 - 1;
    double vel_xy = sqrt(p.vx * p.vx + p.vy*p.vy);

    double bright = map(vel_xy, 0.000006, 0.5, 0, full_color);


    double bright_b = bright >= 0 && bright < color_up_lim ? bright : 10;
    double bright_g = bright >= color_down_lim && bright < color_up_lim*2 ? bright : 10;
    double bright_r = bright >= color_down_lim*2 && bright < color_up_lim*3 ? bright : 10;
    render_pixel(tam_p, x, y, bright_r, bright_g, bright_b);
  }
}

void atualiza_particulas_2(double dt, int n_dim, t_particula** A, t_particula* particulas, long int N_PARTICULAS) {
  int tam_p = 0;
  for(long int i = 0; i < N_PARTICULAS; i++) {
    t_particula p = particulas[i];

    int x = p.x;
    int y = mirror_value(p.y, MIRROR_HEIGHT);

    t_particula pvel = A[(int)mirror_value(p.y, MIRROR_WIDTH)][(int)p.x];

    particulas[i].vx = pvel.vx;
    particulas[i].vy = pvel.vy;

    particulas[i].x += pvel.vx * dt;

    if(particulas[i].x < 1) particulas[i].x = 1;
    if(particulas[i].x > n_dim-1) particulas[i].x = n_dim-1;

    particulas[i].y += pvel.vy * dt;
    if(particulas[i].y < 1) particulas[i].y = 1;
    if(particulas[i].y > n_dim-1) particulas[i].y = n_dim-1;
    
    int novo_x = particulas[i].x;
    int novo_y = mirror_value(particulas[i].y, MIRROR_HEIGHT);

    p.rastro_x[0] = novo_x;
    p.rastro_y[0] = novo_y;
  }

  render_2(A, particulas, N_PARTICULAS);
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

int main(int argc, char** argv) {
  int n_dim = WIDTH, nt = 20000;
  long int N_PARTICULAS = 256;
  double dt = 2.5;//0.0019073486;
  t_particula **A = create_matrix(n_dim, n_dim);

  t_particula* particulas;
  particulas = (t_particula*) malloc(N_PARTICULAS * sizeof(t_particula));

  for(long int i = 0; i < N_PARTICULAS; i++) {
    int sq = (int)sqrt(N_PARTICULAS);
    int posx, posy;

    // posx = WIDTH*0.7;
    // posy = HEIGHT*0.1;
    
    posx = 8 + (i % sq) * (WIDTH-8)/sq;
    posy = 8 + (i / sq) * (HEIGHT-8)/sq;

    // posx = (i / sq) + 2;
    // posy = (i % sq) + 2;

    particulas[i].x = posx;
    particulas[i].y = posy;

    particulas[i].vx = 0;
    particulas[i].vy = 0;

    // printf("posx = %d, posy = %d\n", posx, posy);

    particulas[i].rastro_x = (double*)malloc(sizeof(double)*TAM_RASTRO);
    particulas[i].rastro_y = (double*)malloc(sizeof(double)*TAM_RASTRO);

    particulas[i].rastro_x[0] = posx;
    particulas[i].rastro_y[0] = posy;

    for(int r = 1; r < TAM_RASTRO; r++) {
      particulas[i].rastro_x[r] = 0.0;
      particulas[i].rastro_y[r] = HEIGHT - 0.0;
    }
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
    atualiza_particulas_2(dt, n_dim, A, particulas, N_PARTICULAS);

    // // Pinta os pixels na janela
    // render(A, particulas, N_PARTICULAS);


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