#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #include <sys/time.h>
#include <sys/timeb.h>
#include <GLFW/glfw3.h>

#define N 1000
#define MIN(x,y) (x<y ? x:y)
#define SQR(x) (x*x)
#define START_END(func, name) (printf("Inicio de %s\n", name), func, printf("Fim de %s\n", name))

#define WIDTH 260
#define HEIGHT WIDTH
#define N_PARTICULAS 128

#define REDCT_NUMTHREADS 256

#define NUMBLOCKS 32

#define TAM_BUFFER_SOR 257

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

// GLubyte PixelBuffer[WIDTH * HEIGHT * 3];

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
double xlen; //
double ylen;
int imax;
int jmax;
double delx;
double dely;

//Time Data
double ttime = 0;
double final_time;
double del_time;
double tau; //factor for time step control

//Pressure Data
int max_iter; //max numer of presssure iterations for a time step
int iter; // SOR iter counter
double res; //norm of pressure equation residual
double eps; //stopping tolerance eps for pressure iteration
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
					2 for no-slip conditions,*/
char problem[N];

// struct timeval start, end;
struct timeb start, end, aux_start;
int state = 0;
int n_iter = 5;
__device__ int d_n_iter = 5;

// Recebe um valor pertencente a um intervalo [min,max] e retorna o valor transformado
// para o intervalo [floor,ceil]
double map(double value, double min, double max, double floor, double ceil) {
    return floor + (ceil - floor) * ((value - min) / (max - min));
}

double absf(double a){
	return a < 0 ? -1*a : a;
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
	char* s,*e;
	int num_param = 0;
	fp = fopen(file_name, "r"); // read mode

	if (fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}
	fscanf(fp,"%s", ch);
	while(!feof(fp)){ 
		s = strtok(ch,":");
		switch(num_param){
			case 0:
				xlen = strtod(s,&e);
				break;
			case 1:
				ylen = strtod(s,&e);
				break;
			case 2:
				imax = strtod(s,&e);
				break;
			case 3:
				jmax = strtod(s,&e);
				break;
			case 4:
				delx = strtod(s,&e);
				if(delx == 0){
					delx = xlen/imax;
				}
				break;
			case 5:
				dely = strtod(s,&e);
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

double *vx;
double *vy;
double *p;
double *rhs;
double *F;
double *G;

double *d_vx = NULL;
double *d_vy = NULL;
double *d_p = NULL;
double *d_p_red = NULL;
double *d_p_black = NULL;
double *d_p_prev = NULL;
double *d_partial = NULL;
double *d_p_diff = NULL;

double *d_rhs = NULL;
double *d_F = NULL;
double *d_G = NULL;
double *d_flag = NULL;

double *d_diag_n = NULL;
double *d_diag_s = NULL;
double *d_diag_w = NULL;
double *d_diag_e = NULL;
double *d_diag_p = NULL; 

double *d_maxdiff = NULL;

double *d_vxdiff = NULL;
double *d_vydiff = NULL;
double *d_vxflag = NULL;
double *d_vyflag = NULL;

int *d_res = NULL;

double *dp=NULL;


int n_threads,n_blocos ;

cudaError_t err = cudaSuccess;

texture<double, 1, cudaReadModeElementType> tex;

void write_file(char* output){
	FILE *fp;
	int i, j, idx;
	fp = fopen(output, "w"); // write mode

	if (fp == NULL){
	  perror("Error while opening the file.\n");
	  exit(EXIT_FAILURE);
	}
	
	int milliseconds = (int) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	// fprintf(fp,"Time taken: %lf seconds\n", milliseconds/1000.0);
	// fprintf(fp,"Simulation Time: %.5f seconds\n", ttime);

	for(i = 0; i < imax+2; i++){
		for(j = 0; j < jmax+2; j++){
			idx = i*(imax+2)+j;
			fprintf(fp,"vx[%d][%d]=%.10f\n",i,j,vx[idx]);
			fprintf(fp,"vy[%d][%d]=%.10f\n",i,j,vy[idx]);
			// fprintf(fp,"F[%d][%d]=%.10f\n",i,j,F[idx]);
			// fprintf(fp,"G[%d][%d]=%.10f\n",i,j,G[idx]);
			// fprintf(fp,"p[%d][%d]=%.10f\n",i,j,p[idx]);
		}
		if(i < imax+1) fprintf(fp,"\n");
	}
	fclose(fp);
}

void alocate_vectors_host(){
	vx = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	vy = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	p = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	rhs = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	F = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	G = (double*)malloc((imax+2)*(jmax+2) * sizeof(double));
	
	if(vx == NULL || vy == NULL || p == NULL || rhs == NULL || F == NULL || G == NULL){
		printf("It wasn't possible to alocate memory\n");
		exit(0);
	}
}

void alocate_vectors_device(){	
	size_t size = (imax+2)*(jmax+2) * sizeof(double);
	size_t red_black_size = (imax+2)*(jmax+2)/2 * sizeof(double);
	
	err = cudaMalloc((void **)&d_vx, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector vx (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_vy, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector vy (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_p, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
    
  err = cudaMalloc((void **)&d_p_prev, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device previous vector p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_p_diff, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device diffential vector p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_partial, NUMBLOCKS * sizeof(double));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device partial vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_rhs, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector rhs (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_F, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector F (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_G, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector G (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
    
  err = cudaMalloc((void **)&d_diag_n, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector diag_n (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
    
	err = cudaMalloc((void **)&d_diag_s, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector diag_s (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_diag_w, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector diag_w (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_diag_e, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector diag_e (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_diag_p, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector diag_p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_flag, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_flag (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_vxflag, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_vxflag (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_vyflag, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_vyflag (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
		
	err = cudaMalloc((void **)&d_res, sizeof(int));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer res (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMalloc((void **)&d_maxdiff, sizeof(double));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer maxdiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_vxdiff, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer vxdiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_vydiff, size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer vydiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_p_red, red_black_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer d_p_red (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_p_black, red_black_size);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device pointer d_p_black (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void free_vectors_host(){
	free(vx);
	free(vy);
	free(p);
	free(rhs);
	free(F);
	free(G);
}

void free_vectors_device(){
	
	err = cudaFree(d_vx);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector vx (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_vy);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector vy (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_p);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_p_prev);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device previous vector p (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_rhs);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector rhs (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_F);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector F (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_G);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device vector G (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_res);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer res (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	
	err = cudaFree(d_maxdiff);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer maxdiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_vxdiff);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer vxdiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_vydiff);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer vydiff (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	
	err = cudaFree(d_p_red);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer d_p_red (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	
	err = cudaFree(d_p_black);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free device pointer d_p_black (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
}

void copy_vectors_host_to_device(){
	size_t size = (imax+2)*(jmax+2) * sizeof(double);
	
	err = cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector vx from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector vy from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector p from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(d_rhs, rhs, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector rhs from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(d_F, F, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector F from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector G from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}   
}

void copy_vectors_device_to_host(){
	size_t size = (imax+2)*(jmax+2) * sizeof(double);
	
	err = cudaMemcpy(vx, d_vx, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector vx from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(vy, d_vy, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector vy from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(p, d_p, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector p from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(rhs, d_rhs, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector rhs from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(F, d_F, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector F from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(G, d_G, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector G from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__device__ double d_absf(double x){
	return x < 0 ? -1*x : x;
}

__global__ void init_UVP(int imax, int jmax, double vx_init, double vy_init, double p_init, double* d_vx, double* d_vy, double* d_p, double* d_rhs, double* d_F, double* d_G, double* d_p_red, double* d_p_black){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	if(idx < n){
		d_vx[idx] = vx_init;
		d_vy[idx] = vy_init;
		d_p[idx] = p_init;
		d_rhs[idx] = 0;
		d_F[idx] = 0;
		d_G[idx] = 0;

		if(idx < n/2) {
			d_p_red[idx] = p_init;
			d_p_black[idx] = p_init;
		}
	}
}

__global__ void dt_reductionMax(int imax, int jmax, double* d_partial, double* d_v){
	__shared__ 	double cache [REDCT_NUMTHREADS];
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double temp = 0;
	int n = (imax+2)*(jmax+2);
	int inc = blockDim.x*gridDim.x;
	int i, k;
	for(i = idx; i < n; i += inc){
		if(i > (imax+1) && i < (jmax+1)*(imax+2) && i%(imax+2) > 0 && i%(imax+2) < (imax+1)){
			if(temp < d_absf(d_v[i])){
				temp = d_absf(d_v[i]);
			}
		}
	}
	cache[threadIdx.x] = temp;
	__syncthreads();
	
	for(k = (blockDim.x >> 1); k > 0; k >>= 1){
		if(threadIdx.x < k){
			if(cache[threadIdx.x] < cache[threadIdx.x+k]){
				cache[threadIdx.x] = cache[threadIdx.x+k];
			}
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		d_partial[blockIdx.x] = cache[0];
	}
}

void comp_delt(){
	int i;
	double aux;
	double aux2;
	double partialvx[NUMBLOCKS];
	double partialvy[NUMBLOCKS];
	double maxvx = 0;
	double maxvy = 0;
	if(tau > 0){		
		dt_reductionMax<<<REDCT_NUMTHREADS,NUMBLOCKS>>>(imax, jmax, d_partial, d_vx);

		err = cudaMemcpy(partialvx, d_partial, NUMBLOCKS*sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to copy pointer d_partial from device to host vx (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		dt_reductionMax<<<REDCT_NUMTHREADS,NUMBLOCKS>>>(imax, jmax, d_partial, d_vy);
		
		err = cudaMemcpy(partialvy, d_partial, NUMBLOCKS*sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to copy pointer d_partial from device to host vy (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		for(i = 0; i < NUMBLOCKS; i++){	
			if(partialvx[i] > maxvx) maxvx = partialvx[i];
			if(partialvy[i] > maxvy) maxvy = partialvy[i];
		}

		aux = MIN((delx/maxvx), (dely/maxvy));
		aux2 = MIN(aux, ((Re/2)*(SQR(delx)*SQR(dely)/(SQR(delx)+SQR(dely)))));
		del_time = tau*aux2;
	}
}

__global__ void set_NorthBond(int imax, int jmax, int wN, double* d_vx, double* d_vy){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	switch(wN){
		case 1://free-slip condition
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[idx] = d_vx[imax+2+idx];
				d_vy[imax+2+idx] = 0;
			}
			break;			
		case 2://no-slip condition
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[idx] =- d_vx[imax+2+idx];
				d_vy[imax+2+idx] = 0;
			}	
			break;
		default:
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[idx] = d_vx[imax+2+idx];
				d_vy[imax+2+idx] = 0;
			}
			break;
	}
}

__global__ void set_SouthBond(int imax, int jmax, int wS, double* d_vx, double* d_vy){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	switch(wS){
		case 1: //free-slip condition
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[(jmax+1)*(imax+2)+idx] = d_vx[jmax*(imax+2)+idx];
				d_vy[(jmax)*(imax+2)+idx] = 0;
			}
			break;
		case 2://no-slip condition
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[(jmax+1)*(imax+2)+idx] =- d_vx[jmax*(imax+2)+idx];
				d_vy[(jmax+1)*(imax+2)+idx] = 0;
			}
			break;
		default:
			if(idx < n && idx > 0 && idx < imax+1){
				d_vx[(jmax+1)*(imax+2)+idx] = d_vx[jmax*(imax+2)+idx];
				d_vy[(jmax)*(imax+2)+idx] = 0;
			}
			break;	
	}
}

__global__ void set_WestBond(int imax, int jmax, int wW, double* d_vx, double* d_vy){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	switch(wW){
		case 1: //free-slip condition
			if(idx < n && idx%(imax+2) == 1 && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx-1] = d_vy[idx];
			}
			break;
		case 2://no-slip condition
			if(idx < n && idx%(imax+2) == 1 && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx-1] =- d_vy[idx];
			}			
			break;
		default:
			if(idx < n && idx%(imax+2) == 1 && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx-1] = d_vy[idx];
			}
			break;
	}
}

__global__ void set_EastBond(int imax, int jmax, int wE, double* d_vx, double* d_vy){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	switch(wE){
		case 1: //free-slip condition
			if(idx < n && idx%(imax+2) == (imax+1) && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx] = d_vy[idx-1];
			}
			break;
		case 2://no-slip condition
			if(idx < n && idx%(imax+2) == (imax+1) && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx] =- d_vy[idx-1];
			}
			break;
		default:
				if(idx < n && idx%(imax+2) == (imax+1) && idx > (imax+1) && idx < (n-imax-2)){
				d_vx[idx-1] = 0;
				d_vy[idx] = d_vy[idx-1];
			}
			break;
	}
}

void set_bondCond(){
	set_NorthBond<<< n_blocos, n_threads >>>(imax, jmax, wN, d_vx, d_vy);
	set_SouthBond<<< n_blocos, n_threads >>>(imax, jmax, wS, d_vx, d_vy);
	set_WestBond<<< n_blocos, n_threads >>>(imax, jmax, wW, d_vx, d_vy);
	set_EastBond<<< n_blocos, n_threads >>>(imax, jmax, wE, d_vx, d_vy);
}

__global__ void set_lidDrivenCavityProblem(double lid_vel, int imax, int jmax, double* d_vx){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	if(idx < n && idx > 0 && idx < imax+1){
		d_vx[idx] = 2.0*lid_vel - d_vx[imax+2+idx];
	}
}

__device__ double del_vx_sqr_del_x(double gam, double delx, double vx_C_point, double vx_W_point, double vx_E_point){ //d(vx²)/dx
	double aws;
	aws = (SQR((vx_C_point+vx_E_point)/2) - SQR((vx_W_point+vx_C_point)/2))/delx;
	aws += (gam/delx)*((d_absf(vx_C_point+vx_E_point)/2)*((vx_C_point-vx_E_point)/2) - (d_absf(vx_W_point+vx_C_point)/2)*((vx_W_point-vx_C_point)/2));
	return aws;
}

__device__ double del_vx_vy_del_y(double gam, double dely, double vx_N_point, double vx_C_point, double vx_S_point, double vy_C_point, double vy_E_point, double vy_S_point, double vy_SE_point){ //d(vx*vy)/dy
	double aws;
	aws = ((vy_C_point+vy_E_point)*(vx_C_point+vx_N_point) - (vy_S_point+vy_SE_point)*(vx_S_point+vx_C_point))/2*dely;
	aws += (gam/2*dely)*(d_absf(vy_C_point+vy_E_point)*(vx_C_point-vx_N_point) - d_absf(vy_S_point+vy_SE_point)*(vx_S_point-vx_C_point));
	return aws;
}

__device__ double del_sqr_vx_del_sqr_x(double delx, double vx_E_point, double vx_C_point, double vx_W_point){ //d²(vx)/dx²
	return (vx_E_point-2*vx_C_point+vx_W_point)/SQR(delx);
}

__device__ double del_sqr_vx_del_sqr_y(double dely, double vx_N_point, double vx_C_point, double vx_S_point){ //d²(vx)/dy²
	return (vx_N_point-2*vx_C_point+vx_S_point)/SQR(dely);
}

__device__ double del_vy_sqr_del_y(double gam, double dely, double vy_N_point, double vy_C_point, double vy_S_point){ // d(vy²)/dy
	double aws;
	aws = (SQR((vy_C_point+vy_N_point)/2) - SQR((vy_S_point+vy_C_point)/2))/dely;
	aws += (gam/dely)*(d_absf((vy_C_point+vy_N_point)/2)*((vy_C_point-vy_N_point)/2) - d_absf((vy_S_point+vy_C_point)/2)*((vy_S_point-vy_C_point)/2));
	return aws;
}

__device__ double del_vy_vx_del_x(double gam, double delx, double vy_W_point, double vy_C_point, double vy_E_point, double vx_N_point, double vx_C_point, double vx_W_point, double vx_NW_point){ //d(vx*vy)/dx
	double aws;
	aws = ((vx_C_point+vx_N_point)*(vy_C_point+vy_E_point) - (vx_W_point+vx_NW_point)*(vy_W_point+vy_C_point))/2*delx;
	aws += (gam/2*delx)*(d_absf(vx_C_point+vx_N_point)*(vy_C_point-vy_E_point) - d_absf(vx_W_point+vx_NW_point)*(vy_W_point-vy_C_point));
	return aws;
}

__device__ double del_sqr_vy_del_sqr_x(double delx, double vy_E_point, double vy_C_point, double vy_W_point){//d²(vy)/dx²
	return (vy_E_point-2*vy_C_point+vy_W_point)/SQR(delx);
}

__device__ double del_sqr_vy_del_sqr_y(double dely, double vy_N_point, double vy_C_point, double vy_S_point){//d²(vy)/dy²
	return (vy_N_point-2*vy_C_point+vy_S_point)/SQR(dely);
}

__global__ void comp_FG(int imax, int jmax, double gam, double delx, double dely, double Re, double gx, double gy, double del_time, double* d_vx, double* d_vy, double* d_F, double* d_G){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	
	if(idx < n && idx > (imax+1) && idx < (jmax+1)*(imax+2) && idx%(imax+2) > 0 && idx%(imax+2) < (imax)){
		double term1 = del_sqr_vx_del_sqr_x(delx, d_vx[idx+1], d_vx[idx], d_vx[idx-1]); 
								// del_sqr_vx_del_sqr_x(matrix[j][i+1].vx, matrix[j][i].vx,matrix[j][i-1].vx);
				
		double term2 = del_sqr_vx_del_sqr_y(dely, d_vx[idx-(imax+2)], d_vx[idx], d_vx[idx+(imax+2)]); 
								// del_sqr_vx_del_sqr_y(matrix[j-1][i].vx, matrix[j][i].vx,matrix[j+1][i].vx);
				
		double term3 = del_vx_sqr_del_x(gam, delx, d_vx[idx], d_vx[idx-1], d_vx[idx+1]);
								// del_vx_sqr_del_x(matrix[j][i].vx,matrix[j][i-1].vx,matrix[j][i+1].vx);
				
		double term4 = del_vx_vy_del_y(gam, dely, d_vx[idx-(imax+2)], d_vx[idx], d_vx[idx+(imax+2)], d_vy[idx], d_vy[idx+1], d_vy[idx+(imax+2)], d_vy[idx+(imax+2)+1]); 
								// del_vx_vy_del_y(matrix[j-1][i].vx,matrix[j][i].vx,matrix[j+1][i].vx, matrix[j][i].vy,matrix[j][i+1].vy,matrix[j+1][i].vy,matrix[j+1][i+1].vy);
		
		d_F[idx] = d_vx[idx] + del_time*(((term1+term2)/Re) - term3 - term4 + gx);
		//matrix[j][i].F=matrix[j][i].vx + del_time*(((term1+term2)/Re) - term3 - term4 + gx);	
	
	} else if(idx < n && idx > (imax+1) && idx < (jmax+1)*(imax+2) && (idx%(imax+2) == 0 || idx%(imax+2) == (imax+1)) ) {
		d_F[idx] = d_vx[idx]; //matrix[j][imax].F=matrix[j][imax].vx;
	}
	
	if(idx < n && idx >= 2*(imax+2) && idx < (jmax+1)*(imax+2) && idx%(imax+2) > 0 && idx%(imax+2) < (imax+1)){
		
		double term5 = del_sqr_vy_del_sqr_x(delx, d_vy[idx+1], d_vy[idx], d_vy[idx-1]);		
								// del_sqr_vy_del_sqr_x(matrix[j][i+1].vy,matrix[j][i].vy,matrix[j][i-1].vy);
		
		double term6 = del_sqr_vy_del_sqr_y(dely, d_vy[idx-(imax+2)], d_vy[idx], d_vy[idx+(imax+2)]); 
								// del_sqr_vy_del_sqr_y(matrix[j-1][i].vy,matrix[j][i].vy,matrix[j+1][i].vy);
		
		double term7 = del_vy_vx_del_x(gam, delx, d_vy[idx-1], d_vy[idx], d_vy[idx+1], d_vx[idx-(imax+2)], d_vx[idx], d_vx[idx-1], d_vx[idx-(imax+2)-1]); 
								// del_vy_vx_del_x(matrix[j][i-1].vy, matrix[j][i].vy,matrix[j][i+1].vy, matrix[j-1][i].vx,matrix[j][i].vx, matrix[j][i-1].vx ,matrix[j-1][i-1].vx);
				
		double term8 = del_vy_sqr_del_y(gam, dely, d_vy[idx-(imax+2)], d_vy[idx], d_vy[idx+(imax+2)]);
								// del_vy_sqr_del_y(matrix[j-1][i].vy ,matrix[j][i].vy,matrix[j+1][i].vy);
		
		d_G[idx] = d_vy[idx] + del_time*(((term5+term6)/Re) - term7 - term8 + gy);
	
	} else if(idx < n && idx > (imax+1) && idx%(imax+2) > 0 && idx%(imax+2) < (imax+1)){
		d_G[idx] = d_vy[idx];
	}	
}

__global__ void comp_RHS(int imax, int jmax, double delx, double dely, double del_time, double* d_rhs, double* d_F, double* d_G){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	int n = (imax+2)*(jmax+2);
	
	if(idx < n && idx > (imax+1) && idx < (jmax+1)*(imax+2) && idx%(imax+2) > 0 && idx%(imax+2) < (imax+1)){
		d_rhs[idx] = ((d_F[idx]-d_F[idx-1])/delx +(d_G[idx]-d_G[idx+(imax+2)])/dely)/del_time;
	}	
}

__global__ void build_poisson_system(int jmax, int imax, double delx, double dely, double* d_diag_n, double* d_diag_s, double* d_diag_e, double* d_diag_w, double* d_diag_p, double* d_p_red, double* d_p_black){
	int i, j, idx;
	int ew, ee, es, en;
	for(j = jmax; j > 0; j--){
		es = j<jmax ? 1 : 0;
		en = j>1 ? 1 : 0;
		for(i = 1; i < imax+1; i++){
			idx = j*(imax+2)+i;
			ew = i>1 ? 1 : 0;
			ee = i<imax ? 1 : 0;
			d_diag_e[idx] = ee/SQR(delx);
			d_diag_w[idx] = ew/SQR(delx);
			d_diag_p[idx] = ((ee+ew)/SQR(delx) + (en+es)/SQR(dely));
			d_diag_n[idx] = en/SQR(dely);
			d_diag_s[idx] = es/SQR(dely);
		}
	}
}

__global__ void reductionMax(int imax, int jmax, double* d_partial, double* d_diff, double* d_flag){
	__shared__ 	double cache [REDCT_NUMTHREADS];
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double temp = 0;
	int n = (imax+2)*(jmax+2);
	int inc = blockDim.x*gridDim.x;
	int i, k;
	
	for(i = idx; i < n; i += inc){
		if(d_flag[i]){
			if(temp < d_diff[i]) temp = d_diff[i];
		}
	}
	
	cache[threadIdx.x] = temp;
	__syncthreads();
	
	for(k = (blockDim.x >> 1); k > 0; k >>= 1){
		if(threadIdx.x < k){
			if(cache[threadIdx.x] < cache[threadIdx.x+k]){
				cache[threadIdx.x] = cache[threadIdx.x+k];
			}
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		d_partial[blockIdx.x] = cache[0];
	}
}

__global__ void red_SOR(int imax, int jmax, double omg, double* d_rhs, double* d_p_diff, double* d_diag_n, double* d_diag_s, double* d_diag_e, double* d_diag_w, double* d_diag_p, double* d_flag, double* d_p_red, double* d_p_black){	
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = (imax+2)
	int size = (imax+2)*(jmax+2);
	int new_imax = (imax+2)/2;
	int line = idx / new_imax; // line = 1
	int paridade = line % 2; // paridade = 0
	double aux;
	
	int nidx_red = 2*idx + (idx/new_imax)%2;
	int nidx_black = 2*idx + ((idx/new_imax + 1))%2;

	// __shared__ double d_buffer[TAM_BUFFER_SOR*4];

	// if(threadIdx.x == TAM_BUFFER_SOR*3 - 1) {
	// 	d_buffer[threadIdx.x + TAM_BUFFER_SOR] = d_p_black[idx + new_imax];
	// 	// if(threadIdx.x + TAM_BUFFER_SOR > TAM_BUFFER_SOR*4-1) printf("max = | id =\n");
	// 	if(idx + new_imax > size/2) printf("max = | id =\n");
	// }
	// d_buffer[threadIdx.x] = d_p_black[idx];

	// __syncthreads();

	// if(idx == 0) printf("blockDim.x = %d\n", blockDim.x);
	// idx = j*(imax+2)+i;
	// idx+1 = j*(imax+2)+i+1;
	// idx+1+(imax+2) = j*(imax+2)+(imax+2)+(i+1);
	// idx+1+(imax+2) = (j+1)*(imax+2) +(i+1);

	if(	idx < size/2 && nidx_red < size && nidx_red > (imax+1) && nidx_red < (imax+2)*(jmax+1) && nidx_red % (imax+2) > 0 && nidx_red % (imax+2) < (imax+1)){

		aux = d_diag_s[nidx_red]*d_p_black[idx + new_imax] + d_diag_n[nidx_red]*d_p_black[idx - new_imax]+ d_diag_e[nidx_red]*d_p_black[idx+paridade] + d_diag_w[nidx_red]*d_p_black[idx-1+paridade];
		aux = (1-omg)*d_p_red[idx] + omg*(aux-d_rhs[nidx_red])/d_diag_p[nidx_red];
		d_p_diff[nidx_red] = d_absf(aux-d_p_red[idx]);		
		d_flag[nidx_red] = 1;
		d_p_red[idx] = aux;
	}

	if(	nidx_black < size) d_flag[nidx_black] = 0;
}

__global__ void black_SOR(int imax, int jmax, double omg, double* d_rhs, double* d_p_diff, double* d_diag_n, double* d_diag_s, double* d_diag_e, double* d_diag_w, double* d_diag_p, double* d_flag, double* d_p_red, double* d_p_black){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int size = (imax+2)*(jmax+2);
	int new_imax = (imax+2)/2;
	int line = idx / new_imax;
	int paridade =!(line % 2);
	double aux;
	
	int nidx_black = 2*idx + ((idx/new_imax + 1))%2;
	int nidx_red = 2*idx + (idx/new_imax)%2;

	if(idx < size/2 && nidx_black < size && nidx_black > (imax+1) && nidx_black < (imax+2)*(jmax+1) && nidx_black % (imax+2) > 0 && nidx_black % (imax+2) < (imax+1)){

		aux = d_diag_s[nidx_black]*d_p_red[idx + new_imax] + d_diag_n[nidx_black]*d_p_red[idx - new_imax]+ d_diag_e[nidx_black]*d_p_red[idx+paridade] + d_diag_w[nidx_black]*d_p_red[idx-1+paridade];
		aux = (1-omg)*d_p_black[idx] + omg*(aux-d_rhs[nidx_black])/d_diag_p[nidx_black];
		d_p_diff[nidx_black] = d_absf(aux-d_p_black[idx]);		
		d_flag[nidx_black] = 1;
		d_p_black[idx] = aux;
	}
	
	if(	nidx_red < size) d_flag[nidx_red] = 0;
}

__global__ void check_diff(double* d_partial, double* d_diff, double* diff) {
	__shared__ double diff_local[1];
	diff_local[0] = 0.0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(d_partial[idx] > diff_local[0]) diff_local[0] = d_partial[idx];

	// printf("diff = %lf\n", diff[0]);
	__syncthreads();
	if(idx == 0) {
		*d_diff = diff_local[0];
		// printf("diff_device = %lf\n", diff_local[0]);
	}
}

int calc_interval(void* start, void* end) {
	struct timeb *start_time = (timeb*)start;
	struct timeb *end_time = (timeb*)end;
	ftime(end_time);

	int total_time = (int) (1000.0 * (end_time->time - start_time->time)
			+ (end_time->millitm - start_time->millitm));

	return total_time;
}

int Poisson(){
	int iter = 0;
	int i;
	double partial[NUMBLOCKS];
	double diff = 0;
	struct timeb start_rbsor, end_rbsor, start_redct, end_redct, start_aux, end_aux, start_total, end_total;
	int total_rbsor = 0, total_redct = 0, total_aux = 0, total = 0;

	// int n_blocos_sor = ((imax+2)*(jmax+2)/2)/TAM_BUFFER_SOR*3;
	// printf("%d == %d\n", n_blocos_sor * TAM_BUFFER_SOR, (imax)*(jmax));
	
	ftime(&start_total);
	while(iter < max_iter){
		ftime(&start_rbsor);
		for(i = 0; i < n_iter; i++){
			red_SOR<<<n_blocos, n_threads>>>(imax, jmax, omg, d_rhs, d_p_diff, d_diag_n, d_diag_s, d_diag_e, d_diag_w, d_diag_p, d_flag, d_p_red, d_p_black);
			black_SOR<<<n_blocos, n_threads>>>(imax, jmax, omg, d_rhs, d_p_diff, d_diag_n, d_diag_s, d_diag_e, d_diag_w, d_diag_p, d_flag, d_p_red, d_p_black);
		}

		ftime(&start_redct);
		diff = 0;
		reductionMax<<<REDCT_NUMTHREADS, NUMBLOCKS>>>(imax, jmax, d_partial, d_p_diff, d_flag);
		total_rbsor += calc_interval(&start_rbsor, &end_rbsor);
		
		err = cudaMemcpy(partial, d_partial, NUMBLOCKS*sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to copy pointer d_partial from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// if(iter > 500) {
		// 	iter += n_iter;
		// 	continue;
		// }

		total_redct += calc_interval(&start_redct, &end_redct);
		ftime(&start_aux);

		for(i = 0; i < NUMBLOCKS; i++){
			if(partial[i] > diff) diff = partial[i];
		}
		total_aux += calc_interval(&start_aux, &end_aux);
		iter += n_iter;
		
		if(diff < eps){
			break;
			// return iter;
		}
	}
	// printf("diff = %lf\n", diff);

	ftime(&end_total);
	total += (int) (1000.0 * (end_total.time - start_total.time)
				+ (end_total.millitm - start_total.millitm));

	// printf("Total time - reductionMax: %d milliseconds\n", total_redct);
	// printf("Total time - redBlackSOR: %d milliseconds\n", total_rbsor);
	// printf("Total time - aux: %d milliseconds\n", total_aux);
	// printf("Total time - total: %d milliseconds\n", total);
	return iter;
}

__global__ void d_adap_Vel(int imax, int jmax, double delx, double dely, double del_time, double* d_vx, double* d_vy, double* d_F, double* d_G, double* d_vxdiff, double* d_vydiff, double* d_vxflag, double* d_vyflag, double* d_p_red, double* d_p_black){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = (imax+2)*(jmax+2);
	int line = idx / (imax+2);
	int paridade = line % 2;

	double dp_nidx = idx % 2 == paridade ? d_p_red[idx/2] : d_p_black[idx/2];
	double dp_nidx_i = (idx+1) % 2 == paridade ? d_p_red[(idx+1)/2] : d_p_black[(idx+1)/2];
	double dp_nidx_imax = idx % 2 == paridade ? d_p_black[(idx-(imax+2))/2] : d_p_red[(idx-(imax+2))/2];
	
	double aux;
	if(idx < n && idx > (imax+1) && idx < (jmax+1)*(imax+2) && idx%(imax+2) > 0 && idx%(imax+2) < (imax)){
		aux = d_F[idx] - (del_time*(dp_nidx_i - dp_nidx)/delx); //matrix[j][i].F - (del_time*(matrix[j][i+1].p - matrix[j][i].p)/delx);
		d_vxdiff[idx] = d_absf(aux-d_vx[idx]);   //absf(aux-matrix[j][i].vx);
		d_vxflag[idx] = 1;
		d_vx[idx] = aux;		
	} else{
		d_vxflag[idx] = 0;
	}
	
	if(idx < n && idx >= 2*(imax+2) && idx < (jmax+1)*(imax+2) && idx%(imax+2) > 0 && idx%(imax+2) < (imax+1)){
		aux = d_G[idx] - (del_time*(dp_nidx_imax - dp_nidx)/dely); //matrix[j][i].G - (del_time*(matrix[j-1][i].p - matrix[j][i].p)/dely);
		d_vydiff[idx] = d_absf(aux-d_vy[idx]);//absf(aux-matrix[j][i].vy);
		d_vyflag[idx] = 1;
		d_vy[idx] = aux;
	}	else{
		d_vyflag[idx] = 0;
	}	
}

int adap_Vel(int n_blocos, int n_threads){
	double diffvx = 0;
	double diffvy = 0;
	int i;
	
	double partialvx[NUMBLOCKS];
	double partialvy[NUMBLOCKS];
	
	d_adap_Vel<<< n_blocos, n_threads >>>(imax, jmax, delx, dely, del_time, d_vx, d_vy, d_F, d_G, d_vxdiff, d_vydiff, d_vxflag, d_vyflag, d_p_red, d_p_black);		

	reductionMax<<<REDCT_NUMTHREADS, NUMBLOCKS>>>(imax, jmax, d_partial, d_vxdiff, d_vxflag);
	err = cudaMemcpy(partialvx, d_partial, NUMBLOCKS*sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)	{
		fprintf(stderr, "Failed to copy pointer d_partial from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	reductionMax<<<REDCT_NUMTHREADS, NUMBLOCKS>>>(imax, jmax, d_partial, d_vydiff, d_vyflag);
	err = cudaMemcpy(partialvy, d_partial, NUMBLOCKS*sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)	{
		fprintf(stderr, "Failed to copy pointer d_partial from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
		
	for(i = 0; i < NUMBLOCKS; i++){
		if(partialvy[i] > diffvy) diffvy = partialvy[i];
		if(partialvx[i] > diffvx) diffvx = partialvx[i];
	}
	
	if(diffvy < eps && diffvx < eps) return 1;
	return 0;
}

int main(int argc, char ** argv){
	read_file(argv[1]);
	alocate_vectors_host();
	alocate_vectors_device();
	copy_vectors_host_to_device();
    
	n_threads = (imax+2);
	n_blocos = ((imax+2)*(jmax+2)+ n_threads-1)/n_threads;
	init_UVP<<<n_blocos, n_threads>>>(imax, jmax, vx_init, vy_init, p_init, d_vx, d_vy, d_p, d_rhs, d_F, d_G, d_p_red, d_p_black);
	build_poisson_system<<<1, 1>>>(jmax, imax, delx, dely, d_diag_n, d_diag_s, d_diag_e, d_diag_w, d_diag_p, d_p_red, d_p_black);
	
	cudaArray* cuArray;

	cudaMallocArray(&cuArray, &tex.channelDesc, n_threads*n_threads/2, 1);
	cudaBindTextureToArray(tex, cuArray);

	tex.filterMode = cudaFilterModeLinear;

	int set_time = 1;
	double ant_del_time = 1.0;
	double eps_time = 1e-7;
	int num_time = 0;
	int limit = 100;
	int max_frames = 2000;
	int dyn_eps_flag = 0;

  // gettimeofday(&start, NULL);
	ftime(&start);

	int frame = 0, frames = 0;
	float last_time_frame = 0.0, last_frame_time = 0.0;
	printf("Entrando no laco principal\n");
	while(!state){
		ftime(&aux_start);
		if(set_time){
			comp_delt();
		}
		if(absf(ant_del_time-del_time) < eps_time){
			num_time++;
			if(num_time == limit){
				set_time = 0;
			}
		}	else{
			num_time = 0;
		}
		
		set_bondCond();
		set_lidDrivenCavityProblem<<<n_blocos, n_threads>>>(1.0, imax, jmax, d_vx);
		
		comp_FG<<<n_blocos, n_threads>>>(imax, jmax, gam, delx, dely, Re, gx, gy, del_time, d_vx, d_vy, d_F, d_G);
		comp_RHS<<<n_blocos, n_threads>>>(imax, jmax, delx, dely, del_time, d_rhs, d_F, d_G);		

		Poisson();

		// if(time_frame < 1000) break;
		// printf("Time = %lf/%lf\r", ttime, final_time);
		
		state = adap_Vel(n_blocos, n_threads);
		ftime(&end);
		
		int time_frame = (int) (1000.0 * (end.time - aux_start.time)
        + (end.millitm - aux_start.millitm));
		ttime += del_time;
		ant_del_time = del_time;

		frame++; frames++;
		ftime(&end);
		time_frame = (int) (1000.0 * (end.time - start.time)
        + (end.millitm - start.millitm));

		// printf("Total time elapsed: %lf seconds - Iteracoes: %d\n", time_frame/1000.0, frames);
		// printf("Time = %d\n", time_frame);
		// if(time_frame - last_frame_time > 1000.0/24.0 && eps <= 0.001 && !dyn_eps_flag){
		// 	double range = absf(1000.0/24.0 - (time_frame - last_frame_time));
		// 	// printf("range = %lf\n", log2(range));
		// 	eps *= 2.0;//log(range);
		// 	eps = min(0.001, eps);
		// }
		// if(time_frame - last_frame_time < 1000.0/30.0 && !dyn_eps_flag){
		// 	eps /= 1.1;
		// 	if(eps <= 1e-7) dyn_eps_flag = 1;
		// }
		if(time_frame - last_time_frame >= 1000.0) {
			printf("============ %d fps\r", frame);
			frame = 0;
			last_time_frame = time_frame;
		}

		last_frame_time = time_frame;
		// break;
	}
	
	set_bondCond();		
	set_lidDrivenCavityProblem<<<n_blocos, n_threads>>>(1.0, imax, jmax, d_vx);
	// gettimeofday(&end, NULL);
	ftime(&end);

	int milliseconds = (int) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
	
	// printf("Time = %lf/%lf\n", ttime, final_time);
	printf("Total time elapsed: %lf seconds\n\n", milliseconds/1000.0);

	copy_vectors_device_to_host(); 
  write_file(argv[2]);
    
  free_vectors_device();	
	free_vectors_host();

	return 0;
}
