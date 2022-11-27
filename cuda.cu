#include <stdio.h>
#include <time.h>

#define N 2

// wrapper para checar erros nas chamadas de funções de CUDA
#define CUDA_SAFE_CALL(call){ \
  cudaError_t err = call; \
  if(err != cudaSuccess){ \
    fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }\
}

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
  float *h_a, *d_a;
  float *h_b, *d_b;
  float *h_c, *d_c;
  unsigned long long n_bytes = N * sizeof(float);

  // aloca memória na CPU(host)
  h_a = (float*) malloc(n_bytes);
  h_b = (float*) malloc(n_bytes);
  h_c = (float*) malloc(n_bytes);
  if(h_a == NULL || h_b == NULL || h_c == NULL) exit(EXIT_FAILURE);

  // inicializa os vetores h_a e h_b
  for(int i = 0; i < N; i++) {
    h_a[i] = i+1;
    h_b[i] = i+1;
  }

  // aloca espaço para os vetores na GPU
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, n_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, n_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, n_bytes));

  // copia os vetores da CPU para a GPU (host para device)
  CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice));
  clock_t tempo;
  tempo = clock();
  // Dispara o kernel com N threads
  VecAdd<<<1, N>>>(d_a, d_b, d_c);
  printf("Tempo gasto: %.0lfms\n", (clock() - tempo)*1000.0/CLOCKS_PER_SEC);
  CUDA_SAFE_CALL(cudaGetLastError());

  // copia resultado da GPU para a CPU (device para host)
  CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, n_bytes, cudaMemcpyDeviceToHost));

  for(int i = 0; i < N; i++) {
    float aux = h_a[i] + h_b[i];
    if(aux != h_c[i]){
      printf("Resultado incorreto no índice %d\n", i);
      break;
    }
    if(i == N-1) printf("Resultado correto\n");
  }

  //libera memória na GPU
  CUDA_SAFE_CALL(cudaFree(d_a));
  CUDA_SAFE_CALL(cudaFree(d_b));
  CUDA_SAFE_CALL(cudaFree(d_c));

  return 0;
}