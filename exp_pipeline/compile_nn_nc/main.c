#include "nsnet.h"
#include <stdio.h>
#include <omp.h>

#define NUM_EXPERIMENTS (10000)

void print_output(float32_t* output, int size, const char* path);

int main(int argc, char* argv[]) {
    if(argc != 4) {
        printf("Invalid number of parameters\n");
        return -1;
    }
    
    const int INPUT_SIZE = atoi(argv[1]);
    const int HIDDEN_SIZE = atoi(argv[2]);
    const char* path = argv[3];
    const int OUTPUT_SIZE = INPUT_SIZE;

    float* experiments = (float32_t*) malloc(sizeof(float32_t) * NUM_EXPERIMENTS);

    float32_t* in_noisy = (float32_t*) malloc(sizeof(float32_t) * INPUT_SIZE);
    for(int i=0; i<INPUT_SIZE; i++)
        in_noisy[i] = 1.0f;
    
    float32_t* in_hidden1 = (float32_t*) malloc(sizeof(float32_t) * HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++)
        in_hidden1[i] = 0.0f;
    
    float32_t* in_hidden2 = (float32_t*) malloc(sizeof(float32_t) * HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE; i++)
        in_hidden2[i] = 0.0f;
    
    float32_t* output = (float32_t*) malloc(sizeof(float32_t) * OUTPUT_SIZE);
    float32_t* out_hidden1 = (float32_t*) malloc(sizeof(float32_t) * HIDDEN_SIZE);
    float32_t* out_hidden2 = (float32_t*) malloc(sizeof(float32_t) * HIDDEN_SIZE);
    
    double start_benchmark, end_benchmark;

    allocnn();
    for(int curr_exp=0; curr_exp<NUM_EXPERIMENTS; curr_exp++) {
        start_benchmark = omp_get_wtime();
        run_inference(in_noisy, in_hidden1, in_hidden2, output, out_hidden1, out_hidden2);
        end_benchmark = omp_get_wtime();

        experiments[curr_exp] = end_benchmark - start_benchmark;

        float32_t* temp = out_hidden1;
        out_hidden1 = in_hidden1;
        in_hidden1 = temp;

        temp = out_hidden2;
        out_hidden2 = in_hidden2;
        in_hidden2 = temp;
    }
    freenn();

    print_output(experiments, NUM_EXPERIMENTS, path);

    free(in_noisy);
    free(in_hidden1);
    free(in_hidden2);
    free(out_hidden1);
    free(out_hidden2);
    free(output);
    free(experiments);

    return 0;
}

void print_output(float32_t* output, int size, const char* path) {
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%.10f", output[i]);
        if (i != size - 1) {
            fprintf(file, ";");
        }
    }

    fclose(file);
}