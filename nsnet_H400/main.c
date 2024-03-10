#include "nsnet.h"
#include <stdio.h>

#define INPUT_SIZE (257)
#define OUTPUT_SIZE (INPUT_SIZE)
#define HIDDEN_SIZE (400)

#define NUM_EXPERIMENTS (10000)
#define NUM_MODULES (17)

void print_output(float32_t* output, int size1, int size2);

int main() {
    float* experiments = (float32_t*) malloc(sizeof(float32_t) * NUM_EXPERIMENTS * NUM_MODULES);

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
    
    allocnn();
    for(int curr_exp=0; curr_exp<NUM_EXPERIMENTS; curr_exp++) {
        run_inference(in_noisy, in_hidden1, in_hidden2, output, out_hidden1, out_hidden2, experiments, NUM_EXPERIMENTS, NUM_MODULES, curr_exp);
        
        float32_t* temp = out_hidden1;
        out_hidden1 = in_hidden1;
        in_hidden1 = temp;

        temp = out_hidden2;
        out_hidden2 = in_hidden2;
        in_hidden2 = temp;
    }
    freenn();

    print_output(experiments, NUM_EXPERIMENTS, NUM_MODULES);

    //for(int i=0; i<NUM_EXPERIMENTS; i++) {
    //    for(int j=0; j<NUM_MODULES; j++) {
    //        printf("%.20f ", experiments[i*NUM_MODULES+j]);
    //    }
    //    printf("\n");
    //}

    free(in_noisy);
    free(in_hidden1);
    free(in_hidden2);
    free(out_hidden1);
    free(out_hidden2);
    free(output);
    free(experiments);

    return 0;
}

void print_output(float32_t* output, int size1, int size2) {
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < size1; i++) {
        for(int j=0; j < size2; j++) {
            fprintf(file, "%.10f", output[i*size2+j]);
            if (i*size2+j != size1*size2 - 1) {
                fprintf(file, ";");
            }
        }
    }

    fclose(file);
}