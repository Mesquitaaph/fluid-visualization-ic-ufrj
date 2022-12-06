#ifndef SOLVELU_H_
#define SOLVELU_H_

double** create_matrix(int n, int m);
int mult_matriz_vec(int n, double* A, double* X, double* b);
void solveMatrix(int dim_mat, double** m,  double* b, double* X);

#endif