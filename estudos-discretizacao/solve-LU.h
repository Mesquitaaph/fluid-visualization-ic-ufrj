#ifndef SOLVELU_H_
#define SOLVELU_H_

void DECOMPOSICAO_LU(int n, double** A, double** rL, double** rU);
void SOLVE_LU(int n, double** A, double* b, double* X);

#endif