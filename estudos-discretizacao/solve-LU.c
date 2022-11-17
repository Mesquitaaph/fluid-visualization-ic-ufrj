#include <stdio.h>
#include <stdlib.h>

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
    for(int j = 0; j < i-1; j++) {
      sum += A[i][j]*X[j];
    }

    X[i] = ( b[i] - sum ) / A[i][i];
  }
}


// Abaixo em construção ------------------------------------
/** Função que implementa a decomposição LU em uma matriz nxn.
    Recebe uma matriz A quadrada nxn
    Retorna L e U da decomposição
*/
void DECOMPOSICAO_LU(int n, double** A, double* b, double** L, double** U) {
  Id = Matrix{Float64}(I, n, n) // Matriz identidade nxn
  L = Id // Inicializando a matriz L com a matriz identidade
  U = A // Inicializando a matriz U com a matriz A

  // Função que recebe uma posição i,j, monta uma
  // matriz E de zeros nxn e substitui E[i,j] por 1
  // Retorna a matriz resultante
  double** E(i,j) {
    e = zeros(n, n)
    e[i,j] = 1
    return e
  }

  // Neste laço montamos as matrizes L e U
  // Sendo U o resultado de aplicar a eliminação gaussiana em A
  // e L o inverso da matriz que multiplica A para obter U
  for(i in 1:(n-1)) {
    for(j in (i+1):n) {
      // Acha o fator que multiplica a linha i
      // para fazer o elemento U[j,i] = 0
      k = -U[j][i] / U[i][i]

      // Expressão matricial, na matriz U, equivalente a multiplicar
      // a linha i por k e somar na linha j
      U = (Id + k*E(j,i)) * U

      // Análogo à expressão acima, porém com a diferença de que
      // ao final L, é a inversa da matriz que multiplica A e obtém U
      L = L * (Id - k*E(j,i))
    }
  }
}

int main() {

  return 0;
}