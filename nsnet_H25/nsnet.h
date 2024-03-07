// INCLUDE

#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


typedef float float32_t;

// MACROS

#define NUM_THREADS (4)


#define READMAT(OUT_MATRIX, SIZE, MATRIX_NAME, TYPE) \
    OUT_MATRIX = (TYPE*) malloc(SIZE * sizeof(TYPE)); \
    fp = fopen(MATRIX_NAME, "rb"); \
    if (fp != NULL) { \
        for (int i = 0; i < SIZE; i++) { \
            int res; \
            TYPE x; \
            res = fread(&x, sizeof(TYPE), 1, fp); \
            OUT_MATRIX[i] = x; \
        } \
        fclose(fp); \
    }

#define BENCHMARK(NAME, NFLOPS) \
    printf("BENCHMARK\n"); \
    printf("node name: %s\n", NAME); \
    neuralcasting_end_benchmark = omp_get_wtime(); \
    neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark; \
    printf("time: %f\n", neuralcasting_time_benchmark); \
    printf("nflops: %d\n", NFLOPS); \
    printf("nflops/time: %f\n", NFLOPS/neuralcasting_time_benchmark);

// allocnn
void allocnn();

// freenn
void freenn();

void run_inference(float32_t* tensor_onnxMatMul_0, float32_t* tensor_hidden1, float32_t* tensor_hidden, float32_t* tensor_163, float32_t* tensor_84, float32_t* tensor_148);
