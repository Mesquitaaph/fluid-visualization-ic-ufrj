#include <stdio.h>
#include <stdlib.h>

#define DX 0.01

double f(double x) {
  return x*x*x;
}

double f_l(double x) {
  return 3*x*x;
}

double f_ll(double x) {
  return 6*x;
}

void dominio(double* X, double a, double b) {
  int i = 0;

  for(double x = a; x <= b; x += DX) {
    X[i++] = x;
  }
}

void imagem(int N, double* X, double* Y) {
  for(int i = 0; i < N; i++) {
    Y[i] = f(X[i]);
  }
}

void derivadaSimplesAnterior(int N, double* Y, double* Y_l) {
  for(int i = 1; i < N; i++) {
    double derivada = (Y[i] - Y[i-1]) / DX;
    Y_l[i] = derivada;
  }
}

void derivadaSimplesCentral(int N, double* Y, double* Y_l) {
  for(int i = 1; i < N-1; i++) {
    double derivada = (Y[i+1] - Y[i-1]) / (2*DX);
    Y_l[i] = derivada;
  }
}

void derivadaSegunda(int N, double* Y, double* Y_ll) {
  for(int i = 2; i < N-2; i++) {
    double derivada = (Y[i+1] - 2*Y[i] + Y[i-1]) / (DX*DX);
    Y_ll[i] = derivada;
  }
}

int main() {
  double a = 0;
  double b = 10;

  int N = (b - a) / DX  + 1;

  double *X, *Y, *Y_l, *Y_ll;

  // unsigned long long n_bytes = N * sizeof(double);

  X = (double*) malloc(N * sizeof(double));
  Y = (double*) malloc(N * sizeof(double));
  Y_l = (double*) malloc(N * sizeof(double));
  Y_ll = (double*) malloc(N * sizeof(double));

  dominio(X, a, b);
  imagem(N, X, Y);
  derivadaSimplesCentral(N, Y, Y_l);
  derivadaSegunda(N, Y, Y_ll);

  // for(int i = 0; i < N; i++) {
  //   printf("f(%lf) = %lf\n", X[i], Y[i]);
  // }

  for(int i = 2; i < N-2; i++) {
    printf("f''(%lf) = %lf (%lf)\n", X[i], Y_ll[i], f_ll(X[i]));
  }

  return 0;
}