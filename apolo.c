#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 50
#define MIN_SIZE 999
#define MAX_SIZE 1000
#define MAX_NUM 10000000.0

int main() {
    srand((unsigned) time(NULL));
    int size = 2000, iterator = 0;
    double time_spent = 0.0;

    // Punteros para las matrices
    double (*A)[size];
    double (*B)[size];
    double (*C)[size];

    // Calculate the total size of the matrices
    size_t total_size = size * size * sizeof(double);

    // Asigna la memoria para las matrices alineadas usando aligned_alloc
    A = aligned_alloc(64, total_size);
    B = aligned_alloc(64, total_size);
    C = aligned_alloc(64, total_size);

    if (A == NULL || B == NULL || C == NULL) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    while (iterator++ < N) {
        printf("Running iteration number: %d\n", iterator);

        clock_t begin_time = clock();

        // Paralelizar la inicialización de las matrices
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                A[j][i] = 1 + ((double)rand() / RAND_MAX) * MAX_NUM;
                B[i][j] = 1 + ((double)rand() / RAND_MAX) * MAX_NUM;
                C[j][i] = 0;
            }
        }

        // Paralelizar la multiplicación de matrices
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

    // Free the allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
