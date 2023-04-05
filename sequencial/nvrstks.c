#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <GLFW/glfw3.h>

#define N 1000
#define MIN(x,y) (x<y ? x:y)
#define SQR(x) (x*x)

#define WIDTH 256
#define HEIGHT WIDTH
#define N_PARTICULAS 128

typedef struct Particula {
  double x, y, vx, vy;
} t_particula;

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

// Atribui ao pixel na posição (x,y) a cor [r,g,b]
void makePixel(int x, int y, int r, int g, int b, GLubyte* pixels, int width, int height) {
  if (0 <= x && x < width && 0 <= y && y < height) {
    int position = (x + y * width) * 3;
    pixels[position] = r;
    pixels[position + 1] = g;
    pixels[position + 2] = b;
  }
}

//Spacial Data
double xlen;
double ylen;
int imax;
int jmax;
double delx;
double dely;

//Time Data
double timee=0;
double final_time;
double del_time;
double tau; //factor for time step control

//Pressure Data
int max_iter; //max numer of presssure iterations for a time step
int iter; // SOR iter counter
double res; //norm of pressure equation residual
double eps; //stopping tolerance eps for pressure iteration
double eps_stability = 0.00001;
double omg; //relaxation parameter u> for SOR iteration
double gam; //upwind differencing factor 


//Problem dependent Data
double Re;
double gx;
double gy;
double vx_init;
double vy_init;
double p_init;
int wW,wE,wN,wS; /*specify the type of boundary condition along the
					western (left), eastern (right), northern (upper), and
					southern (lower) boundaries of 17 = [0,xlength] x
					[0,ylength]; each may have one of the values:
					1 for free-slip conditions,
					2 for no-slip conditions*/
char problem[N];
struct timeval start, end;

double absf(double a){
	return a < 0 ? -1*a : a;
}

typedef struct Point {
   double vx;
   double vy;
   double p;
   double rhs;
   double F;
   double G;
} POINT;

typedef struct Pair{
	double p1;
	double p2;
} PAIR;

POINT** matrix;

// Recebe um valor pertencente a um intervalo [min,max] e retorna o valor transformado
// para o intervalo [floor,ceil]
double map(double value, double min, double max, double floor, double ceil) {
    return floor + (ceil - floor) * ((value - min) / (max - min));
}

void render(double time, t_particula *particulas) {
  int tam_p = 0;
	double bright = map(time, 0, final_time, 16, 255);

  for(int z = 0; z < N_PARTICULAS; z++) {
    t_particula p = particulas[z];
    int x = p.x;
    int y = p.y;

    for(int i = -tam_p; i <= tam_p; i++) {
      for(int j = -tam_p; j <= tam_p; j++) {
        makePixel(
          x+i, y+j, 
          bright, bright, bright,
          PixelBuffer, WIDTH, HEIGHT
        );
      }
    }
  }
}

void atualiza_particulas(double dt, /*int n_dim,*/ POINT** velocidades, t_particula* particulas) {
  for(int i = 0; i < N_PARTICULAS; i++) {
    t_particula p = particulas[i];

    int x = p.x;
    int y = p.y;

    // Limpa o espaco em que a particula estava
    // for(int i = -1; i <= 1; i++) {
    //   for(int j = -1; j <= 1; j++) {
    //     makePixel(
    //       x+i, y+j, 
    //       0, 0, 0,
    //       PixelBuffer, WIDTH, HEIGHT
    //     );
    //   }
    // }

    POINT pvel = velocidades[WIDTH-(int)p.y][(int)p.x];

    particulas[i].vx = pvel.vx;
    particulas[i].vy = pvel.vy;

    particulas[i].x += pvel.vx * dt;
    // if(particulas[i].x < 1 || particulas[i].x > n_dim-1) particulas[i].x = 0;

    particulas[i].y += pvel.vy * dt;
    // if(particulas[i].y < 1 || particulas[i].y > n_dim-1) particulas[i].y = 0;
  }
}

void print_values(){
	printf("Printing values:\n");
	printf("xlen=%f\n",xlen);
	printf("ylen=%f\n",ylen);
	printf("imax=%d\n",imax);
	printf("jmax=%d\n",jmax);
	printf("delx=%f\n",delx);
	printf("dely=%f\n",dely);
	printf("final_time=%f\n",final_time);
	printf("del_time=%f\n",del_time);
	printf("tau=%f\n",tau);
	printf("max_iter=%d\n",max_iter);
	printf("res=%f\n",res);
	printf("eps=%f\n",eps);
	printf("omg=%f\n",omg);
	printf("gam=%f\n",gam);
	printf("Re=%f\n",Re);
	printf("gx=%f\n",gx);
	printf("gy=%f\n",gy);
	printf("vx_init=%f\n",vx_init);
	printf("vy_init=%f\n",vy_init);
	printf("p_init=%f\n",p_init);
	printf("wW=%d\n",wW);
	printf("wE=%d\n",wE);
	printf("wN=%d\n",wW);
	printf("wS=%d\n",wS);
	printf("problem=%s\n",problem);
	printf("-------------------------------\n\n");
}
void read_file(char * file_name){
	FILE *fp;
	char ch[50];
	char* s, *e;
	int num_param = 0;
	fp = fopen(file_name, "r"); // read mode

	if (fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}

	fscanf(fp, "%s", ch);
	while(!feof(fp)){ 
		s = strtok(ch, ":");
		switch(num_param){
			case 0:
				xlen = strtod(s, &e);
				break;
			case 1:
				ylen = strtod(s, &e);
				break;
			case 2:
				imax = strtod(s, &e);
				break;
			case 3:
				jmax = strtod(s, &e);
				break;
			case 4:
				delx = strtod(s, &e);
				if(delx == 0){
					delx = xlen/imax;
				}
				break;
			case 5:
				dely = strtod(s, &e);
				if(dely == 0){
					dely = ylen/jmax;
				}
				break;
			case 6:
				final_time = strtod(s,&e);
				break;
			case 7:
				del_time = strtod(s,&e);
				break;
			case 8:
				tau = strtod(s,&e);
				break;
			case 9:
				max_iter = strtod(s,&e);
				break;
			case 10:
				res = strtod(s,&e);
				break;
			case 11:
				eps = strtod(s,&e);
				break;
			case 12:
				omg = strtod(s,&e);
				break;
			case 13:
				gam = strtod(s,&e);
				break;
			case 14:
				Re = strtod(s,&e);
				break;
			case 15:
				gx = strtod(s,&e);
				break;
			case 16:
				gy = strtod(s,&e);
				break;
			case 17:
				vx_init = strtod(s,&e);
				break;
			case 18:
				vy_init = strtod(s,&e);
				break;
			case 19:
				p_init = strtod(s,&e);
				break;
			case 20:
				wW = strtod(s,&e);
				break;
			case 21:
				wE = strtod(s,&e);
				break;
			case 22:
				wN = strtod(s,&e);
				break;
			case 23:
				wS = strtod(s,&e);
				break;
			case 24:
				strcpy(problem,s);
				break;	
			default:			
				exit(666);
				break;
			}
		num_param++;
		fscanf(fp,"%s", ch);
	}
	fclose(fp);
}

void write_file(char * output){
	FILE *fp;
	int i, j;
	fp = fopen(output, "w"); // write mode

	if(fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}
	// fprintf(fp,"Time taken: %ld seconds\n", end.tv_sec-start.tv_sec);
	fprintf(fp, "Simulation Time: %.5f seconds\n", timee);
	for(i = 0; i < imax+2; i++){
		for(j = 0; j < jmax+2; j++){			
			fprintf(fp, "vx[%d][%d]=%.10f\n", i, j, matrix[i][j].vx);
			fprintf(fp, "vy[%d][%d]=%.10f\n", i, j, matrix[i][j].vy);
			// fprintf(fp,"p[%d][%d]=%f\n",i,j,matrix[i][j].p);
			// fprintf(fp,"rhs[%d][%d]=%f\n",i,j,matrix[i][j].rhs);
			// fprintf(fp,"F[%d][%d]=%f\n",i,j,matrix[i][j].F);
			// fprintf(fp,"G[%d][%d]=%.10f\n",i,j,matrix[i][j].G);
		}
		fprintf(fp, "\n");
	}
	
	fclose(fp);
}

void clear_file(char * output) {
	FILE* fp;
	fp = fopen(output, "w"); // write mode
	fclose(fp);
}

void write_state_file(char * output){
	FILE *fp;
	int i, j;
	fp = fopen(output, "a"); // write mode

	if(fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}

	for(i = 0; i < imax+2; i++){
		for(j = 0; j < jmax+2; j++){			
			fprintf(fp, "vx[%d][%d]=%.10f\n", i, j, matrix[i][j].vx);
			fprintf(fp, "vy[%d][%d]=%.10f\n", i, j, matrix[i][j].vy);
		}
	}
	fprintf(fp, "\n");
	
	fclose(fp);
}

void alocate_matrix(){
	matrix = (POINT **)malloc((imax + 2)*sizeof(POINT*));
	if(matrix == NULL){
		printf("It wasn't possible to alocate memory\n");
		exit(0);
	}
	
	for(int i = 0; i < imax+2; i++) {
		matrix[i] = (POINT *)malloc((jmax+2) * sizeof(POINT));
		if(matrix[i]==NULL){
			printf("It wasn't possible to alocate memory\n");
			exit(0);
		}
	}
}

void free_matrix(){
	for(int i = 0; i < imax; i++) {
		free(matrix[i]);
	}
	free(matrix);
}

void init_UVP(){
	int i, j;
	for(i = 0; i < imax+2; i++){
		for(j = 0; j < jmax+2; j++){
			matrix[i][j].vx = vx_init;
			matrix[i][j].vy = vy_init;
			matrix[i][j].p = p_init;
			matrix[i][j].rhs = 0;
			matrix[i][j].F = 0;
			matrix[i][j].G = 0;			
		}
	}
}

void print_matrix(){
	int i, j;
	for(i = 0; i < imax; i++){
		for(j = 0; j < jmax; j++){			
			printf("vx[%d][%d] = %f\n", i, j, matrix[i][j].vx);
			printf("vy[%d][%d] = %f\n", i, j, matrix[i][j].vy);
			printf("p[%d][%d] = %f\n", i, j, matrix[i][j].p);
			printf("rhs[%d][%d] = %f\n", i, j, matrix[i][j].rhs);
			printf("F[%d][%d] = %f\n", i, j, matrix[i][j].F);
			printf("G[%d][%d] = %f\n", i, j, matrix[i][j].G);
			printf("\n");
		}
		printf("\n");
	}
}

PAIR find_max_vel(){
	PAIR pair;
	int i, j;
	pair.p1 = absf(matrix[1][1].vx);
	pair.p2 = absf(matrix[1][1].vy);
	for(i = 1; i < imax+1; i++){
		for(j = 1; j < jmax+1; j++){			
			if(absf(matrix[i][j].vx) > pair.p1){
				pair.p1 = absf(matrix[i][j].vx);
			}
			if(absf(matrix[i][j].vy) > pair.p2){
				pair.p2 = absf(matrix[i][j].vy);
			}
		}
	}
	return pair;
}

void comp_delt(){
	PAIR pair;
	double aux;
	double aux2;
	if(tau > 0) {
		pair = find_max_vel();
		aux =  MIN((delx/pair.p1), (dely/pair.p2));
		aux2 = MIN(aux, ((Re/2)*(SQR(delx)*SQR(dely)/(SQR(delx) + SQR(dely)))));
		del_time = tau*aux2;
	}	
}

void set_NorthBond(){
	int i;
	switch(wN){
		case 1://free-slip condition
			for(i = 1; i < imax+1; i++){
				matrix[0][i].vx = matrix[1][i].vx;
				matrix[1][i].vy = 0;
			}
			break;
		case 2://no-slip condition
			for(i = 1; i < imax+1; i++){
				matrix[0][i].vx =- matrix[1][i].vx;
				matrix[1][i].vy = 0;				
			}
			break;
		default:
			for(i = 1; i < imax+1; i++){
				matrix[0][i].vx = matrix[1][i].vx;
				matrix[1][i].vy = 0;
			}
			break;
	}
}

void set_SouthBond(){
	int i;
	switch(wS){
		case 1: //free-slip condition
			for(i = 1; i < imax+1; i++){
				matrix[jmax+1][i].vx = matrix[imax][i].vx;
				matrix[jmax][i].vy = 0;				
			}
			break;
		case 2://no-slip condition
			for(i = 1; i < imax+1; i++){
				matrix[jmax+1][i].vx =- matrix[jmax][i].vx;
				matrix[jmax+1][i].vy = 0;				
			}
			break;
		default:
			for(i = 1; i < imax+1; i++){
				matrix[imax+1][i].vx = matrix[imax][i].vx;
				matrix[imax+1][i].vy = 0;				
			}
			break;;
	
	}
}

void set_WestBond(){
	int j;
	switch(wW){
		case 1: //free-slip condition
			for(j = 1; j < jmax+1; j++){
				matrix[j][0].vx = 0;
				matrix[j][0].vy = matrix[j][1].vy;
			}
			break;
		case 2://no-slip condition
			for(j = 1; j < jmax+1; j++){
				matrix[j][0].vx = 0;				
				matrix[j][0].vy =- matrix[j][1].vy;
			}
			break;
		default:
			for(j = 1; j < jmax+1; j++){
				matrix[j][0].vx = 0;
				matrix[j][0].vy = matrix[j][1].vy;
			}
			break;
	}
}

void set_EastBond(){
	int j;
	switch(wE){
		case 1: //free-slip condition
			for(j = 1; j < jmax+1; j++){
				matrix[j][imax].vx = 0;
				matrix[j][imax+1].vy = matrix[j][imax].vy;
			}
			break;
		case 2://no-slip condition
			for(j = 1; j < jmax+1; j++){
				matrix[j][imax].vx = 0;			
				matrix[j][imax+1].vy =- matrix[j][imax].vy;
			}
			break;
		default:
			for(j = 1; j < jmax+1; j++){
				matrix[j][imax].vx = 0;
				matrix[j][imax+1].vy = matrix[j][imax].vy;
			}
			break;
	}
}

void set_lidDrivenCavityProblem(double lid_vel){
	int i;
	for(i = 1; i < imax+1; i++){
		matrix[imax+1][i].vx = 2.0*lid_vel - matrix[imax][i].vx;
	}
}

void set_bondCond(){
	set_NorthBond();
	set_SouthBond();
	set_WestBond();
	set_EastBond();
}


double del_vx_sqr_del_x(double vx_C_point, double vx_W_point, double vx_E_point){ //d(vx²)/dx
	double aws;
	
	aws = (SQR((vx_C_point + vx_E_point)/2) - SQR((vx_W_point + vx_C_point)/2))/delx;
	aws += (gam/delx)*((absf(vx_C_point + vx_E_point)/2)*((vx_C_point - vx_E_point)/2) - (absf(vx_W_point + vx_C_point)/2)*((vx_W_point - vx_C_point)/2));

	return aws;
}

double del_vx_vy_del_y(double vx_N_point, double vx_C_point, double vx_S_point, double vy_C_point, double vy_E_point, double vy_S_point, double vy_SE_point){ //d(vx*vy)/dy
	double aws;
	
	aws = ((vy_C_point + vy_E_point)*(vx_C_point + vx_N_point) - (vy_S_point + vy_SE_point)*(vx_S_point + vx_C_point))/2*dely;
	aws += (gam/2*dely)*(absf(vy_C_point + vy_E_point)*(vx_C_point - vx_N_point) - absf(vy_S_point + vy_SE_point)*(vx_S_point - vx_C_point));

	return aws;
}

double del_sqr_vx_del_sqr_x(double vx_E_point, double vx_C_point, double vx_W_point){ //d²(vx)/dx²
	return (vx_E_point - 2*vx_C_point + vx_W_point)/SQR(delx);
}

double del_sqr_vx_del_sqr_y(double vx_N_point, double vx_C_point, double vx_S_point){ //d²(vx)/dy²
	return (vx_N_point - 2*vx_C_point + vx_S_point)/SQR(dely);
}

double del_vy_sqr_del_y(double vy_N_point, double vy_C_point, double vy_S_point){ // d(vy²)/dy
	double aws;
	
	aws = (SQR((vy_C_point+vy_N_point)/2) - SQR((vy_S_point+vy_C_point)/2))/dely;
	aws += (gam/dely)*(absf((vy_C_point+vy_N_point)/2)*((vy_C_point-vy_N_point)/2) - absf((vy_S_point+vy_C_point)/2)*((vy_S_point-vy_C_point)/2));
	
	return aws;
}

double del_vy_vx_del_x(double vy_W_point, double vy_C_point, double vy_E_point, double vx_N_point, double vx_C_point, double vx_W_point, double vx_NW_point){ //d(vx*vy)/dx
	double aws;
	
	aws = ((vx_C_point + vx_N_point)*(vy_C_point + vy_E_point) - (vx_W_point + vx_NW_point)*(vy_W_point + vy_C_point))/2*delx;
	aws += (gam/2*delx)*(absf(vx_C_point + vx_N_point)*(vy_C_point - vy_E_point) - absf(vx_W_point + vx_NW_point)*(vy_W_point - vy_C_point));
	return aws;
}

double del_sqr_vy_del_sqr_x(double vy_E_point, double vy_C_point, double vy_W_point){//d²(vy)/dx²
	return (vy_E_point - 2*vy_C_point + vy_W_point)/SQR(delx);
}
double del_sqr_vy_del_sqr_y(double vy_N_point, double vy_C_point, double vy_S_point){//d²(vy)/dy²
	return (vy_N_point - 2*vy_C_point + vy_S_point)/SQR(dely);
}

void comp_FG(){
	int i, j;
	for(j = jmax; j > 0; j--){
		for(i = 1; i < imax+1; i++){
			if(i < imax){
				double term1 = del_sqr_vx_del_sqr_x(matrix[j][i+1].vx, matrix[j][i].vx, matrix[j][i-1].vx);
				double term2 = del_sqr_vx_del_sqr_y(matrix[j-1][i].vx, matrix[j][i].vx, matrix[j+1][i].vx);

				double term3 = del_vx_sqr_del_x(matrix[j][i].vx, matrix[j][i-1].vx, matrix[j][i+1].vx);

				double term4 = del_vx_vy_del_y(matrix[j-1][i].vx, matrix[j][i].vx, matrix[j+1][i].vx, matrix[j][i].vy, matrix[j][i+1].vy, matrix[j+1][i].vy, matrix[j+1][i+1].vy);

				matrix[j][i].F = matrix[j][i].vx + del_time*(((term1 + term2)/Re) - term3 - term4 + gx);
			}
			if(j > 1){
				double term5 = del_sqr_vy_del_sqr_x(matrix[j][i+1].vy, matrix[j][i].vy, matrix[j][i-1].vy);				
				double term6 = del_sqr_vy_del_sqr_y(matrix[j-1][i].vy, matrix[j][i].vy, matrix[j+1][i].vy);
				
				double term7 = del_vy_vx_del_x(matrix[j][i-1].vy, matrix[j][i].vy, matrix[j][i+1].vy, matrix[j-1][i].vx, matrix[j][i].vx, matrix[j][i-1].vx, matrix[j-1][i-1].vx);
				
				double term8 = del_vy_sqr_del_y(matrix[j-1][i].vy, matrix[j][i].vy, matrix[j+1][i].vy);
				
				matrix[j][i].G = matrix[j][i].vy + del_time*(((term5 + term6)/Re) - term7 - term8 + gy);
			}

			matrix[jmax+1][i].G = matrix[jmax+1][i].vy;
			matrix[1][i].G = matrix[1][i].vy;			
		}

		matrix[j][0].F = matrix[j][0].vx;
		matrix[j][imax].F = matrix[j][imax].vx;
	}
}

void comp_RHS(){
	int i, j;
	for(j = jmax; j > 0; j--){
		for(i = 1; i < imax+1; i++){
			matrix[j][i].rhs = ((matrix[j][i].F - matrix[j][i-1].F)/delx + (matrix[j][i].G - matrix[j+1][i].G)/dely)/del_time;
		}
	}
}

int Poisson(){
	int iter = 0;
	int i, j;
	double aux;
	int ew, ee, es, en;
	double max;
	double diff;
	while(iter < max_iter){
    // printf("%cIteracao: %d", iter == 0 ? '\0' : '\r', iter);
		max = 0;
		for(i = 1; i < imax+1; i++){
				ew = i > 1 ? 1 : 0;
				ee = i < imax ? 1 : 0;	
			for(j = jmax; j > 0; j--){						
				es = j < jmax ? 1 : 0;
				en = j > 1 ? 1 : 0;			
				aux = omg/((ee + ew)/SQR(delx) + (en + es)/SQR(dely));				
				aux = aux * ((ee*matrix[j][i+1].p + ew*matrix[j][i-1].p)/SQR(delx) + (en*matrix[j-1][i].p + es*matrix[j+1][i].p)/SQR(dely) - matrix[j][i].rhs); 
				aux = (1 - omg)*matrix[j][i].p + aux;
				diff = absf(matrix[j][i].p - aux);
				if(diff > max) max = diff;
				matrix[j][i].p = aux;
			}
		}

		iter++;
		if(max < eps) return iter;		
	}

	return iter;
}

int adap_Vel(){
	int i, j;
	double aux, diffvx, diffvy, max_vx = 0, max_vy = 0;
	for(j = jmax; j > 0; j--){
		for(i = 1; i < imax+1; i++){
			if(i < imax){
				aux = matrix[j][i].F - (del_time*(matrix[j][i+1].p - matrix[j][i].p)/delx);
				diffvx = absf(aux-matrix[j][i].vx);
				matrix[j][i].vx = aux;
				if( diffvx > max_vx) max_vx = diffvx;				
			}
			if(j > 1){
				aux = matrix[j][i].G - (del_time*(matrix[j-1][i].p - matrix[j][i].p)/dely);
				diffvy = absf(aux - matrix[j][i].vy);
				matrix[j][i].vy = aux;
				if(diffvy > max_vy) max_vy = diffvy;
			}
		}
	}	
	if(max_vy < eps && max_vx < eps) return 1;

	return 0;
}

int main(int argc, char** argv) {
	
	int set_time = 0;
	double ant_del_time = 0;
	int num_time = 0;
	int limit_time = 100;
	int state = 0;

	read_file(argv[1]);
	alocate_matrix();
	init_UVP();
	gettimeofday(&start, NULL);
	clear_file(argv[2]);

	t_particula particulas[N_PARTICULAS];
  for(int i = 0; i < N_PARTICULAS; i++) {
    int sq = (int)sqrt(N_PARTICULAS);

    particulas[i].x = 2+(i % sq) * (WIDTH-2)/sq;
    particulas[i].y = 2+(i / sq) * (HEIGHT-2)/sq;
    particulas[i].vx = 0;
    particulas[i].vy = 0;
  }

  GLFWwindow* window;
  // Inicializando a biblioteca
  if (!glfwInit())
    return -1;

  // Criando a janela e seu contexto OpenGL
  window = glfwCreateWindow(WIDTH, HEIGHT, "Visualizacao de movimento de particulas dada equações de Navier-Stokes", NULL, NULL);
  if (!window){
    glfwTerminate();
    return -1;
  }

  // Cria o contexto atual da janela
  glfwMakeContextCurrent(window);

	int frame = 1;
	while(!state){
		// write_state_file(argv[2]);

		// Configuração da visualização
    glfwGetFramebufferSize(window, &WindowMatrixPlot.width, &WindowMatrixPlot.height);
    glViewport(0, 0, WindowMatrixPlot.width, WindowMatrixPlot.height);

    // Atualiza a posicao de cada uma das particulas
    atualiza_particulas(ant_del_time, matrix, particulas);

    // Pinta os pixels na janela
    render(timee, particulas);

		frame++;
		glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);

    // Funções necesárias para o funcionamento da biblioteca que desenha os pixels
    glfwSwapBuffers(window);
    glfwPollEvents();

		if(timee < final_time){
			comp_delt();
			ant_del_time = del_time;	
		}

		if((ant_del_time - del_time) < eps_stability && !set_time){
			num_time++;
			if(num_time == limit_time) set_time = 1;
		} else {
			num_time = 0;
		}
		
		set_bondCond();
		set_lidDrivenCavityProblem(1.0);

		comp_FG();
		comp_RHS();

		Poisson();
    printf("Time = %lf/%lf\r", timee, final_time);

		state = adap_Vel();
		timee += del_time;
		// break;
	}
	set_bondCond();
	set_lidDrivenCavityProblem(1.0);
	
	write_file(argv[2]);
	gettimeofday(&end, NULL);
	printf("delta = %.10lf\n", del_time);
	printf("delta = %.10lf\n", ant_del_time);
	int initSim;
  scanf("%c", &initSim);

	// write_file(argv[2]);
	free_matrix();

	printf("\n");
	return 0;
}
 