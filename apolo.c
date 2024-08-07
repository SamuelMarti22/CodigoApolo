#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 50
#define MIN_SIZE 2
#define MAX_SIZE 5000
#define MAX_NUM 10000000.0

int main() {
    srand((unsigned) time(NULL));
    int size =  MIN_SIZE + (rand() % MAX_SIZE), iterator = 0;
    double time_spent = 0.0;

    // Pointers for the matrix
    double A[size][size] __attribute__((aligned(64))), B[size][size] __attribute__((aligned(64))), C[size][size] __attribute__((aligned(64)));


    while (iterator++ < N) {
        printf("Running iteration number: %d\n", iterator);

        clock_t begin_time = clock();

        // Parallelize matrix initialization
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                A[j][i] = 1 + ((double)rand() / RAND_MAX) * MAX_NUM;
                B[i][j] = 1 + ((double)rand() / RAND_MAX) * MAX_NUM;
                C[j][i] = 0;
            }
        }

        // Parallelize matrix multiplication
        #pragma omp parallel for
        for(int j = 0; j < size/2; j++){
            for(int k = 0; k < size; k++){
                for(int i = 0; i < size; i++){
                    C[i][j] += A[k][j] * B[k][j];
                }
            }
        }

        clock_t end_time = clock();
        time_spent += (double)(end_time - begin_time) / CLOCKS_PER_SEC;
    }

    printf("Size of matrices: %d \n", size);
    printf("Running time: %f \n", (time_spent / N));

    return 0;
}
