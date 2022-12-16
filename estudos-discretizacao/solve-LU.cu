#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "solve-LU.h"

#define N 1000
#define TAM_BLOCO 20

// wrapper para checar erros nas chamadas de funções de CUDA
#define CUDA_SAFE_CALL(call) { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }\
}

__global__ void lu_calc_col(double * d_m, int dim, int i) {
  __shared__ double a_ii;

  if(threadIdx.x == 0) {
    a_ii = d_m[i*(dim+1)];
  }
  __syncthreads();

  int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
  if(j < dim) {
    d_m[j*dim + i] /= a_ii;
  }
}

__global__ void lu_calc_subm(double* d_m, int dim, int i) {
  __shared__ double a_ji[TAM_BLOCO];
  __shared__ double a_ik[TAM_BLOCO];
  int j = blockDim.x * blockIdx.x + threadIdx.x + i + 1;
  int k = blockDim.y * blockIdx.y + threadIdx.y + i + 1;
  if((threadIdx.y == 0) && (j < dim)) {
    if(threadIdx.x >= TAM_BLOCO) printf("idx.x = %u\n", threadIdx.x);
    a_ji[threadIdx.x] = d_m[j*dim + i];
  }
  if((threadIdx.x == 0) && ( k < dim ) ) {
    if(threadIdx.y >= TAM_BLOCO) printf("idx.y = %u\n", threadIdx.y);
    a_ik [ threadIdx.y ] = d_m [ i * dim + k ];
  }
  __syncthreads();
  if((j < dim) && (k < dim)) {
    if(threadIdx.x >= TAM_BLOCO) printf("2idx.x = %u\n", threadIdx.x);
    if(threadIdx.y >= TAM_BLOCO) printf("2idx.y = %u\n", threadIdx.y);
    d_m[j*dim + k] -= a_ji[threadIdx.x] * a_ik[threadIdx.y];
  }
}

void alg_lu_gpu(double* d_m, int dim) {
  int i, n_blocos;
  for(i = 0; i < dim-1; i++) {
    n_blocos = ((dim-i-1) + TAM_BLOCO-1) / TAM_BLOCO;
    
    lu_calc_col <<<n_blocos, TAM_BLOCO>>>(d_m, dim, i);
    CUDA_SAFE_CALL(cudaGetLastError());

    // printf("nblocos = %d\n", n_blocos);
    dim3 g_blocos(n_blocos, n_blocos);
    dim3 n_threads(TAM_BLOCO, TAM_BLOCO);

    // printf("gbx = %u, gby = %u\n", g_blocos.x, g_blocos.y);
    // printf("ntx = %u, nty = %u\n", n_threads.x, n_threads.y);

    lu_calc_subm <<<g_blocos, n_threads>>>(d_m, dim, i);
    CUDA_SAFE_CALL(cudaGetLastError());
  }
}

// Função que resolve um sistema triangular superior AX = b
// Recebe uma matriz quadrada A e um vetor b
// Retorna o vetor X, solução do sistema triangular superior AX = b
void RESOLVE_TRIANGULAR_SUPERIOR(int n, double* A, double* b, double* X) {
  for(int i = 0; i < n; i++) {
    int Xn = n-1-i;

    double sum = 0;
    for(int j = Xn+1; j < n; j++) {
      sum += A[Xn*n + j]*X[j];
    }

    X[Xn] = ( b[Xn] - sum ) / A[Xn*n + Xn];
  }
}

// Função que resolve um sistema triangular inferior AX = b
// Recebe uma matriz quadrada A e um vetor b
// Retorna o vetor X, solução do sistema triangular inferior AX = b
void RESOLVE_TRIANGULAR_INFERIOR(int n, double* A, double* b, double* X) {
  for(int i = 0; i < n; i++) {
    double sum = 0;
    for(int j = 0; j < i; j++) {
      sum += A[i*n + j]*X[j];
    }

    X[i] = (b[i] - sum)/A[i*n + i];
  }
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

void DECOMP_LU_SEQ(int n, double* A, double* rL, double* rU) {
  double **Ad, **rLd, **rUd;
  Ad = (double**)malloc(n*sizeof(double*));
  rLd = (double**)malloc(n*sizeof(double*));
  rUd = (double**)malloc(n*sizeof(double*));

  for(int i = 0; i < n; i++) {
    Ad[i] = (double*)malloc(n*sizeof(double));
    rLd[i] = (double*)malloc(n*sizeof(double));
    rUd[i] = (double*)malloc(n*sizeof(double));

    for(int j = 0; j < n; j++) {
      Ad[i][j] = A[i*n + j];
    }
  }

  DECOMPOSICAO_LU(n, Ad, rLd, rUd);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      rL[i*n + j] = rLd[i][j];
      rU[i*n + j] = rUd[i][j];
    }

    free(Ad[i]);
    free(rLd[i]);
    free(rUd[i]);
  }
  free(Ad);
  free(rLd);
  free(rUd);
}

// Função que recebe a matriz A quadrada nxn e o vetor b nx1 do sistema AX = b
// e obtém a solução X utilizando a decomposição LU, retornando-a.
void SOLVE_LU(int n, double* b, double* X, double* L, double* U) {
  double *c = (double*) malloc(n*sizeof(double));

  // Resolve o sistema Lc = b
  RESOLVE_TRIANGULAR_INFERIOR(n, L, b, c);

  // Resolve o sistema UX = c
  RESOLVE_TRIANGULAR_SUPERIOR(n, U, c, X);
}

/*
  Funcao com o algoritmo de multiplicacao de matrizes tradicional.
*/
int mult_matriz_vec(int n, double* A, double* X, double* b) {
  for(int i = 0; i < n; i++) {
    b[i] = 0;
    for(int k = 0; k < n; k++) {
      b[i] += A[i*n + k] * X[k];
    }
  }
  return 1;
}

void solveMatrix(int dim_mat, double** m, double* b, double* X) {
  double *src_m, *d_m, *L, *U;

  // adicionar c´odigo para inicializar a vari´avel
  // dim_mat ( dimens~ao da matriz )
  unsigned long long quant_mem = dim_mat * dim_mat * sizeof(double);
  src_m = (double*) malloc(quant_mem);
  L = (double*) malloc(quant_mem);
  U = (double*) malloc(quant_mem);
  if(L == NULL || U == NULL) {
    fprintf(stderr, "Memoria insuficiente\n");
    exit(EXIT_FAILURE);
  }

  for(int i = 0; i < dim_mat; i++){
    for(int j = 0; j < dim_mat; j++){
      src_m[i*dim_mat + j] = m[i][j];
    }
  }

  // alocar mem´oria na GPU para copiar a matriz
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_m, quant_mem));

  // copiar a matriz para a GPU
  CUDA_SAFE_CALL(cudaMemcpy(d_m, src_m, quant_mem, cudaMemcpyHostToDevice));

  alg_lu_gpu(d_m, dim_mat);

  // copiar o resultado da GPU para a CPU
  CUDA_SAFE_CALL(cudaMemcpy(src_m, d_m, quant_mem, cudaMemcpyDeviceToHost));

  // limpar a mem´oria da GPU
  CUDA_SAFE_CALL(cudaFree(d_m));

  for(int i = 0; i < dim_mat; i++) {
    for(int j = 0; j < dim_mat; j++) {
      if(j >= i) {
        U[i*dim_mat + j] = src_m[i*dim_mat + j];
      } else {
        U[i*dim_mat + j] = 0.0;
      }
    }
  }

  for(int i = 0; i < dim_mat; i++) {
    for(int j = 0; j < dim_mat; j++) {
      if(j < i) {
        L[i*dim_mat + j] = src_m[i*dim_mat + j];
      } else if(j == i) {
        L[i*dim_mat + j] = 1.0;
      } else {
        L[i*dim_mat + j] = 0.0;
      }
    }
  }
  SOLVE_LU(dim_mat, b, X, L, U);
  free(src_m); free(L); free(U);
}

// int main (int argc, char *argv[]) {
//   int dim_mat = atoi(argv[1]);
//   double *m, *d_m, *L, *U, *Ls, *Us, *b, *X, *Xs, *Xref;

//   // adicionar c´odigo para inicializar a vari´avel
//   // dim_mat ( dimens~ao da matriz )
//   unsigned long long quant_mem = dim_mat * dim_mat * sizeof(double);
//   m = (double*) malloc(quant_mem);
//   L = (double*) malloc(quant_mem);
//   U = (double*) malloc(quant_mem);
//   Ls = (double*) malloc(quant_mem);
//   Us = (double*) malloc(quant_mem);
//   b = (double*) malloc(dim_mat * sizeof(double));
//   X = (double*) malloc(dim_mat * sizeof(double));
//   Xs = (double*) malloc(dim_mat * sizeof(double));
//   Xref = (double*) malloc(dim_mat * sizeof(double));
//   if(m == NULL || L == NULL || U == NULL || b == NULL || X == NULL) {
//     fprintf(stderr, "Memoria insuficiente\n");
//     exit(EXIT_FAILURE);
//   }

//   // adicionar c´odigo para preencher a matriz
//   // e criar outros dados necess´arios para seu problema
//   // alocar mem´oria na GPU para copiar a matriz

//   // Atribuindo valores aleatorios entre -1000 e 1000 para as matrizes A e B.
//   for(int i = 0; i < dim_mat; i++) {
//     for(int j = 0; j < dim_mat; j++) {
//       m[i*dim_mat + j] = (rand() % 2001) - 1000;
//       // printf("%lf ", m[i*dim_mat + j]);
//     }
//   }
//   // printf("\n\n");

//   for(int i = 0; i < dim_mat; i++) {
//     Xref[i] = (rand() % 2001) - 1000;
//   }

//   mult_matriz_vec(dim_mat, m, Xref, b);

//   // alocar mem´oria na GPU para copiar a matriz
//   CUDA_SAFE_CALL(cudaMalloc((void**)&d_m, quant_mem));

//   // copiar a matriz para a GPU
//   CUDA_SAFE_CALL(cudaMemcpy(d_m, m, quant_mem, cudaMemcpyHostToDevice));

//   // Decomposição LU sequencial
//   clock_t tempoSeq;
//   tempoSeq = clock();
//   DECOMP_LU_SEQ(dim_mat, m, Ls, Us);
//   printf("Tempo gasto sequencial: %.0lfms\n", (clock() - tempoSeq)*1000.0/CLOCKS_PER_SEC);

//   // Decomposição LU concorrente
//   clock_t tempoConc;
//   tempoConc = clock();
//   // executar a fatora¸c~ao na GPU
//   alg_lu_gpu(d_m, dim_mat);
//   printf("Tempo gasto concorrente: %.0lfms\n", (clock() - tempoConc)*1000.0/CLOCKS_PER_SEC);

//   // copiar o resultado da GPU para a CPU
//   CUDA_SAFE_CALL(cudaMemcpy(m, d_m, quant_mem, cudaMemcpyDeviceToHost));

//   // limpar a mem´oria da GPU
//   CUDA_SAFE_CALL(cudaFree(d_m));

//   // adicionar c´odigo para usar
//   // as matrizes L e U ( contidas em m)
//   //e os outros dados
//   for(int i = 0; i < dim_mat; i++) {
//     for(int j = 0; j < dim_mat; j++) {
//       if(j >= i) {
//         U[i*dim_mat + j] = m[i*dim_mat + j];
//       } else {
//         U[i*dim_mat + j] = 0.0;
//       }
//     }
//   }

//   for(int i = 0; i < dim_mat; i++) {
//     for(int j = 0; j < dim_mat; j++) {
//       if(j < i) {
//         L[i*dim_mat + j] = m[i*dim_mat + j];
//       } else if(j == i) {
//         L[i*dim_mat + j] = 1.0;
//       } else {
//         L[i*dim_mat + j] = 0.0;
//       }
//     }
//   }

//   SOLVE_LU(dim_mat, b, X, L, U);
//   SOLVE_LU(dim_mat, b, Xs, Ls, Us);


//   printf("Vector X = \n");
//   int teste = 1;
//   for(int i = 0; i < dim_mat; i++) {
//     if(Xref[i] - Xs[i] < 1/10/10/10/10/10/10/10/10 || Xs[i] - Xref[i] < 1/10/10/10/10/10/10/10/10) {
//         continue;
//     } else {
//       teste = 0;
//       printf("Xref = %lf, Xs = %lf\n", Xref[i], Xs[i]);
//       break;
//     }
//   }
//   printf("%s\n\n", teste ? "Deu certo sequencial" : "Deu ruim sequencial");

//   printf("Vector X = \n");
//   teste = 1;
//   for(int i = 0; i < dim_mat; i++) {
//     if(Xref[i] - X[i] < 1/10/10/10/10/10/10/10/10 || X[i] - Xref[i] < 1/10/10/10/10/10/10/10/10) {
//         continue;
//     } else {
//       teste = 0;
//       printf("Xref = %lf, X = %lf\n", Xref[i], X[i]);
//       break;
//     }
//   }
//   printf("%s\n\n", teste ? "Deu certo concorrente" : "Deu ruim concorrente");

//   free(m);
//   CUDA_SAFE_CALL(cudaDeviceReset());
//   exit(EXIT_SUCCESS);
// }

