// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-03-07 15:52:10
// *****************************************************************************

#include "nsnet.h"

// matrices delaration
static float32_t* tensor_fc1bias;
static float32_t* tensor_fc3bias;
static float32_t* tensor_fc4bias;
static float32_t* tensor_fc2bias;
static float32_t* tensor_onnxMatMul_164;
static float32_t* tensor_onnxGRU_182;
static float32_t* tensor_onnxGRU_183;
static float32_t* tensor_onnxGRU_184;
static float32_t* tensor_onnxGRU_202;
static float32_t* tensor_onnxGRU_203;
static float32_t* tensor_onnxGRU_204;
static float32_t* tensor_onnxMatMul_205;
static float32_t* tensor_onnxMatMul_206;
static float32_t* tensor_onnxMatMul_207;

void allocnn() {
FILE *fp;
READMAT(tensor_fc1bias, 100, "fc1bias.bin", float32_t)
READMAT(tensor_fc3bias, 600, "fc3bias.bin", float32_t)
READMAT(tensor_fc4bias, 600, "fc4bias.bin", float32_t)
READMAT(tensor_fc2bias, 257, "fc2bias.bin", float32_t)
READMAT(tensor_onnxMatMul_164, 25700, "onnxMatMul_164.bin", float32_t)
READMAT(tensor_onnxGRU_182, 30000, "onnxGRU_182.bin", float32_t)
READMAT(tensor_onnxGRU_183, 30000, "onnxGRU_183.bin", float32_t)
READMAT(tensor_onnxGRU_184, 600, "onnxGRU_184.bin", float32_t)
READMAT(tensor_onnxGRU_202, 30000, "onnxGRU_202.bin", float32_t)
READMAT(tensor_onnxGRU_203, 30000, "onnxGRU_203.bin", float32_t)
READMAT(tensor_onnxGRU_204, 600, "onnxGRU_204.bin", float32_t)
READMAT(tensor_onnxMatMul_205, 60000, "onnxMatMul_205.bin", float32_t)
READMAT(tensor_onnxMatMul_206, 360000, "onnxMatMul_206.bin", float32_t)
READMAT(tensor_onnxMatMul_207, 154200, "onnxMatMul_207.bin", float32_t)
}

void freenn() {
free(tensor_fc1bias);
free(tensor_fc3bias);
free(tensor_fc4bias);
free(tensor_fc2bias);
free(tensor_onnxMatMul_164);
free(tensor_onnxGRU_182);
free(tensor_onnxGRU_183);
free(tensor_onnxGRU_184);
free(tensor_onnxGRU_202);
free(tensor_onnxGRU_203);
free(tensor_onnxGRU_204);
free(tensor_onnxMatMul_205);
free(tensor_onnxMatMul_206);
free(tensor_onnxMatMul_207);
}

void run_inference(float32_t* tensor_onnxMatMul_0, float32_t* tensor_hidden1, float32_t* tensor_hidden, float32_t* tensor_163, float32_t* tensor_84, float32_t* tensor_148, float32_t* experiments, int num_experiments, int num_modules, int curr_exp) {
omp_set_num_threads(4);

#ifdef COMPILER_BENCHMARK
double neuralcasting_time_benchmark = 0.0f;
double neuralcasting_end_benchmark = 0.0f;
double neuralcasting_start_benchmark = 0.0f;
#endif




// MATMUL OPERATOR fc1MatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc1MatMul_output_0[1 * 100];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#pragma omp parallel for shared(tensor_onnxMatMul_0, tensor_onnxMatMul_164, tensor_fc1MatMul_output_0) collapse(2)
for(int32_t i=0; i<1; i++) {
    for(int32_t j=0; j<100; j++) {
        float32_t temp = 0.0f;
#pragma omp reduction(temp: +)
        for(int32_t k=0; k<257; k++) {
            int32_t index1 = i*257+k;
            int32_t index2 = k*100+j;
            temp += tensor_onnxMatMul_0[index1] * tensor_onnxMatMul_164[index2];
        }
        tensor_fc1MatMul_output_0[i*100 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC1MATMUL] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc1MatMul -----------------\n");

for(int i=0; i<1; i++) {
    for(int j=0; j<100; j++) {
        printf("%f, ", tensor_fc1MatMul_output_0[i*100+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION fc1Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc1Add_output_0[100];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc1bias, tensor_fc1MatMul_output_0, tensor_fc1Add_output_0) private(i0, i1)
for(int i2=0; i2<100; i2++) {

tensor_fc1Add_output_0[i1*100 + i2*1] = tensor_fc1bias[i2*1] + tensor_fc1MatMul_output_0[i1*100 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC1ADD] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc1Add -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_fc1Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TRANSPOSE /gru1/Transpose

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru1Transpose_output_0[100];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
for(int i2=0; i2<100; i2++) {

tensor_gru1Transpose_output_0[i0*(1*100)+i2*(1)] = tensor_fc1Add_output_0[i1*100 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU1TRANSPOSE] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/Transpose -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_gru1Transpose_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// GRU OPERATOR /gru1/GRU

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru1GRU_output_0[100];
#undef CONNECTED_OUTPUT
#endif



#ifdef CONNECTED_HIDDEN
float tensor_84[100];
#undef CONNECTED_HIDDEN
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

{
    // a = W_ir @ x + b_ir
    float a[100];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        a[i] = 0.0f;
#pragma omp reduction(a: +)
        for(int j=0; j<100; j++) {
            a[i] += tensor_onnxGRU_182[(100*100) + i*100 + j] * tensor_gru1Transpose_output_0[j];
        }
        a[i] += tensor_onnxGRU_184[100 + i];
    }

    // b = W_hr @ h1 + b_hr
    float b[100];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        b[i] = 0.0f;
#pragma omp reduction(b: +)
        for(int j=0; j<100; j++) {
            b[i] += tensor_onnxGRU_183[(100*100) + i*100 + j] * tensor_hidden1[j];
        }
        b[i] += tensor_onnxGRU_184[4*100 + i];
    }

    // c = W_iz @ x + b_iz
    float c[100];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        c[i] = 0.0f;
#pragma omp reduction(c: +)
        for(int j=0; j<100; j++) {
            c[i] += tensor_onnxGRU_182[i*100 + j] * tensor_gru1Transpose_output_0[j];
        }
        c[i] += tensor_onnxGRU_184[i];
    }

    // d = W_hz @ h1 + b_hz
    float d[100];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        d[i] = 0.0f;
#pragma omp reduction(d: +)
        for(int j=0; j<100; j++) {
            d[i] += tensor_onnxGRU_183[i*100 + j] * tensor_hidden1[j];
        }
        d[i] += tensor_onnxGRU_184[3*100 + i];
    }

    // e = W_in @ x + b_in
    float e[100];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        e[i] = 0.0f;
#pragma omp reduction(e: +)
        for(int j=0; j<100; j++) {
            e[i] += tensor_onnxGRU_182[(2*100*100) + i*100 + j] * tensor_gru1Transpose_output_0[j];
        }
        e[i] += tensor_onnxGRU_184[2*100 + i];
    }

    // f = W_hn @ h1 + b_hn
    float f[100];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 100; i++) {
        f[i] = 0.0f;
#pragma omp reduction(f: +)
        for(int j=0; j<100; j++) {
            f[i] += tensor_onnxGRU_183[(2*100*100) + i*100 + j] * tensor_hidden1[j];
        }
        f[i] += tensor_onnxGRU_184[5*100 + i];
    }

    // r = sigmoid(a + b)
    float r[100];
#pragma omp parallel for shared(a, b, r)
    for(int i=0; i<100; i++) {
        float s = a[i] + b[i];
        r[i] = 1.0f / (1.0f + expf(-s));
    }

    // z = sigmoid(c + d)
    float z[100];
#pragma omp parallel for shared(c, d, z)
    for(int i=0; i<100; i++) {
        float s = c[i] + d[i];
        z[i] = 1.0f / (1.0f + expf(-s));
    }

    // n = tanh(e + r*f)
    float n[100];
#pragma omp parallel for shared(n, e, r, f)
    for(int i=0; i<100; i++) {
        n[i] = tanh(e[i] + r[i] * f[i]);
    }

    // hn = (1-z) * n + z * h1
    float hn[100];
#pragma omp parallel for shared(tensor_84, z, n, tensor_hidden1, tensor_gru1GRU_output_0)
    for(int i=0; i<100; i++) {
        tensor_84[i] = (1.0f - z[i]) * n[i] + z[i] * tensor_hidden1[i];
        tensor_gru1GRU_output_0[i] = tensor_84[i];
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU1GRU] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/GRU -----------------\n");

for(int i=0; i<100; i++) {
    printf("%f, ", tensor_gru1GRU_output_0[i]);
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// SQUEEZE OPERATOR /gru1/Squeeze

#define CONNECTED_OUTPUT

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#ifdef CONNECTED_OUTPUT
float* tensor_gru1Squeeze_output_0 = tensor_gru1GRU_output_0;
#undef CONNECTED_OUTPUT
#else

#pragma omp parallel for shared(tensor_gru1GRU_output_0, tensor_gru1Squeeze_output_0)
for(int i=0; i<100; i++) {
    tensor_gru1Squeeze_output_0[i] = tensor_gru1GRU_output_0[i];
}
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU1SQUEEZE] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/Squeeze -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_gru1Squeeze_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// GRU OPERATOR /gru2/GRU

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru2GRU_output_0[100];
#undef CONNECTED_OUTPUT
#endif



#ifdef CONNECTED_HIDDEN
float tensor_148[100];
#undef CONNECTED_HIDDEN
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

{
    // a = W_ir @ x + b_ir
    float a[100];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        a[i] = 0.0f;
#pragma omp reduction(a: +)
        for(int j=0; j<100; j++) {
            a[i] += tensor_onnxGRU_202[(100*100) + i*100 + j] * tensor_gru1Squeeze_output_0[j];
        }
        a[i] += tensor_onnxGRU_204[100 + i];
    }

    // b = W_hr @ h1 + b_hr
    float b[100];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        b[i] = 0.0f;
#pragma omp reduction(b: +)
        for(int j=0; j<100; j++) {
            b[i] += tensor_onnxGRU_203[(100*100) + i*100 + j] * tensor_hidden[j];
        }
        b[i] += tensor_onnxGRU_204[4*100 + i];
    }

    // c = W_iz @ x + b_iz
    float c[100];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        c[i] = 0.0f;
#pragma omp reduction(c: +)
        for(int j=0; j<100; j++) {
            c[i] += tensor_onnxGRU_202[i*100 + j] * tensor_gru1Squeeze_output_0[j];
        }
        c[i] += tensor_onnxGRU_204[i];
    }

    // d = W_hz @ h1 + b_hz
    float d[100];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        d[i] = 0.0f;
#pragma omp reduction(d: +)
        for(int j=0; j<100; j++) {
            d[i] += tensor_onnxGRU_203[i*100 + j] * tensor_hidden[j];
        }
        d[i] += tensor_onnxGRU_204[3*100 + i];
    }

    // e = W_in @ x + b_in
    float e[100];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        e[i] = 0.0f;
#pragma omp reduction(e: +)
        for(int j=0; j<100; j++) {
            e[i] += tensor_onnxGRU_202[(2*100*100) + i*100 + j] * tensor_gru1Squeeze_output_0[j];
        }
        e[i] += tensor_onnxGRU_204[2*100 + i];
    }

    // f = W_hn @ h1 + b_hn
    float f[100];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 100; i++) {
        f[i] = 0.0f;
#pragma omp reduction(f: +)
        for(int j=0; j<100; j++) {
            f[i] += tensor_onnxGRU_203[(2*100*100) + i*100 + j] * tensor_hidden[j];
        }
        f[i] += tensor_onnxGRU_204[5*100 + i];
    }

    // r = sigmoid(a + b)
    float r[100];
#pragma omp parallel for shared(a, b, r)
    for(int i=0; i<100; i++) {
        float s = a[i] + b[i];
        r[i] = 1.0f / (1.0f + expf(-s));
    }

    // z = sigmoid(c + d)
    float z[100];
#pragma omp parallel for shared(c, d, z)
    for(int i=0; i<100; i++) {
        float s = c[i] + d[i];
        z[i] = 1.0f / (1.0f + expf(-s));
    }

    // n = tanh(e + r*f)
    float n[100];
#pragma omp parallel for shared(n, e, r, f)
    for(int i=0; i<100; i++) {
        n[i] = tanh(e[i] + r[i] * f[i]);
    }

    // hn = (1-z) * n + z * h1
    float hn[100];
#pragma omp parallel for shared(tensor_148, z, n, tensor_hidden, tensor_gru2GRU_output_0)
    for(int i=0; i<100; i++) {
        tensor_148[i] = (1.0f - z[i]) * n[i] + z[i] * tensor_hidden[i];
        tensor_gru2GRU_output_0[i] = tensor_148[i];
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU2GRU] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/GRU -----------------\n");

for(int i=0; i<100; i++) {
    printf("%f, ", tensor_gru2GRU_output_0[i]);
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// SQUEEZE OPERATOR /gru2/Squeeze

#define CONNECTED_OUTPUT

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#ifdef CONNECTED_OUTPUT
float* tensor_gru2Squeeze_output_0 = tensor_gru2GRU_output_0;
#undef CONNECTED_OUTPUT
#else

#pragma omp parallel for shared(tensor_gru2GRU_output_0, tensor_gru2Squeeze_output_0)
for(int i=0; i<100; i++) {
    tensor_gru2Squeeze_output_0[i] = tensor_gru2GRU_output_0[i];
}
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU2SQUEEZE] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/Squeeze -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_gru2Squeeze_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TRANSPOSE /gru2/Transpose

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru2Transpose_output_0[100];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
for(int i2=0; i2<100; i2++) {

tensor_gru2Transpose_output_0[i0*(1*100)+i2*(1)] = tensor_gru2Squeeze_output_0[i1*100 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + GRU2TRANSPOSE] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/Transpose -----------------\n");
for(int i=0; i<100; i++) {
    printf("%f ", tensor_gru2Transpose_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// MATMUL OPERATOR fc3MatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc3MatMul_output_0[1 * 600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#pragma omp parallel for shared(tensor_gru2Transpose_output_0, tensor_onnxMatMul_205, tensor_fc3MatMul_output_0) collapse(2)
for(int32_t i=0; i<1; i++) {
    for(int32_t j=0; j<600; j++) {
        float32_t temp = 0.0f;
#pragma omp reduction(temp: +)
        for(int32_t k=0; k<100; k++) {
            int32_t index1 = i*100+k;
            int32_t index2 = k*600+j;
            temp += tensor_gru2Transpose_output_0[index1] * tensor_onnxMatMul_205[index2];
        }
        tensor_fc3MatMul_output_0[i*600 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC3MATMUL] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc3MatMul -----------------\n");

for(int i=0; i<1; i++) {
    for(int j=0; j<600; j++) {
        printf("%f, ", tensor_fc3MatMul_output_0[i*600+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION fc3Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc3Add_output_0[600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc3bias, tensor_fc3MatMul_output_0, tensor_fc3Add_output_0) private(i0, i1)
for(int i2=0; i2<600; i2++) {

tensor_fc3Add_output_0[i1*600 + i2*1] = tensor_fc3bias[i2*1] + tensor_fc3MatMul_output_0[i1*600 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC3ADD] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc3Add -----------------\n");
for(int i=0; i<600; i++) {
    printf("%f ", tensor_fc3Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// RELU OPERATOR reluRelu

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_reluRelu_output_0[600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc3Add_output_0, tensor_reluRelu_output_0) private(i0, i1)
for(int i2=0; i2<600; i2++) {

tensor_reluRelu_output_0[i1*600 + i2*1] = tensor_fc3Add_output_0[i1*600 + i2*1] > 0.0f ? tensor_fc3Add_output_0[i1*600 + i2*1] : 0.0f;
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + RELURELU] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT reluRelu -----------------\n");
for(int i=0; i<600; i++) {
    printf("%f ", tensor_reluRelu_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// MATMUL OPERATOR fc4MatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc4MatMul_output_0[1 * 600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#pragma omp parallel for shared(tensor_reluRelu_output_0, tensor_onnxMatMul_206, tensor_fc4MatMul_output_0) collapse(2)
for(int32_t i=0; i<1; i++) {
    for(int32_t j=0; j<600; j++) {
        float32_t temp = 0.0f;
#pragma omp reduction(temp: +)
        for(int32_t k=0; k<600; k++) {
            int32_t index1 = i*600+k;
            int32_t index2 = k*600+j;
            temp += tensor_reluRelu_output_0[index1] * tensor_onnxMatMul_206[index2];
        }
        tensor_fc4MatMul_output_0[i*600 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC4MATMUL] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc4MatMul -----------------\n");

for(int i=0; i<1; i++) {
    for(int j=0; j<600; j++) {
        printf("%f, ", tensor_fc4MatMul_output_0[i*600+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION fc4Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc4Add_output_0[600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc4bias, tensor_fc4MatMul_output_0, tensor_fc4Add_output_0) private(i0, i1)
for(int i2=0; i2<600; i2++) {

tensor_fc4Add_output_0[i1*600 + i2*1] = tensor_fc4bias[i2*1] + tensor_fc4MatMul_output_0[i1*600 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC4ADD] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc4Add -----------------\n");
for(int i=0; i<600; i++) {
    printf("%f ", tensor_fc4Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// RELU OPERATOR relu_1Relu

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_relu_1Relu_output_0[600];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc4Add_output_0, tensor_relu_1Relu_output_0) private(i0, i1)
for(int i2=0; i2<600; i2++) {

tensor_relu_1Relu_output_0[i1*600 + i2*1] = tensor_fc4Add_output_0[i1*600 + i2*1] > 0.0f ? tensor_fc4Add_output_0[i1*600 + i2*1] : 0.0f;
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + RELU1RELU] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT relu_1Relu -----------------\n");
for(int i=0; i<600; i++) {
    printf("%f ", tensor_relu_1Relu_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif


// MATMUL OPERATOR fc2MatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc2MatMul_output_0[1 * 257];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#pragma omp parallel for shared(tensor_relu_1Relu_output_0, tensor_onnxMatMul_207, tensor_fc2MatMul_output_0) collapse(2)
for(int32_t i=0; i<1; i++) {
    for(int32_t j=0; j<257; j++) {
        float32_t temp = 0.0f;
#pragma omp reduction(temp: +)
        for(int32_t k=0; k<600; k++) {
            int32_t index1 = i*600+k;
            int32_t index2 = k*257+j;
            temp += tensor_relu_1Relu_output_0[index1] * tensor_onnxMatMul_207[index2];
        }
        tensor_fc2MatMul_output_0[i*257 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC2MATMUL] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc2MatMul -----------------\n");

for(int i=0; i<1; i++) {
    for(int j=0; j<257; j++) {
        printf("%f, ", tensor_fc2MatMul_output_0[i*257+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION fc2Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc2Add_output_0[257];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc2bias, tensor_fc2MatMul_output_0, tensor_fc2Add_output_0) private(i0, i1)
for(int i2=0; i2<257; i2++) {

tensor_fc2Add_output_0[i1*257 + i2*1] = tensor_fc2bias[i2*1] + tensor_fc2MatMul_output_0[i1*257 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + FC2ADD] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc2Add -----------------\n");
for(int i=0; i<257; i++) {
    printf("%f ", tensor_fc2Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// SIGMOID OPERATOR sigmoidSigmoid



#ifdef CONNECTED_OUTPUT
float32_t tensor_163[257];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc2Add_output_0, tensor_163) private(i0, i1)
for(int i2=0; i2<257; i2++) {

float32_t ex = exp(tensor_fc2Add_output_0[i1*257 + i2*1]);
tensor_163[i1*257 + i2*1] = ex / (1.0f + ex);
}
}
}


#ifdef COMPILER_BENCHMARK
neuralcasting_end_benchmark = omp_get_wtime();
neuralcasting_time_benchmark = neuralcasting_end_benchmark - neuralcasting_start_benchmark;
experiments[curr_exp * num_modules + SIGMOIDSIGMOID] = neuralcasting_time_benchmark;
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT sigmoidSigmoid -----------------\n");
for(int i=0; i<257; i++) {
    printf("%f ", tensor_163[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
}