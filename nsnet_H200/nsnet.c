// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-03-07 15:54:19
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
READMAT(tensor_fc1bias, 200, "fc1bias.bin", float32_t)
READMAT(tensor_fc3bias, 600, "fc3bias.bin", float32_t)
READMAT(tensor_fc4bias, 600, "fc4bias.bin", float32_t)
READMAT(tensor_fc2bias, 257, "fc2bias.bin", float32_t)
READMAT(tensor_onnxMatMul_164, 51400, "onnxMatMul_164.bin", float32_t)
READMAT(tensor_onnxGRU_182, 120000, "onnxGRU_182.bin", float32_t)
READMAT(tensor_onnxGRU_183, 120000, "onnxGRU_183.bin", float32_t)
READMAT(tensor_onnxGRU_184, 1200, "onnxGRU_184.bin", float32_t)
READMAT(tensor_onnxGRU_202, 120000, "onnxGRU_202.bin", float32_t)
READMAT(tensor_onnxGRU_203, 120000, "onnxGRU_203.bin", float32_t)
READMAT(tensor_onnxGRU_204, 1200, "onnxGRU_204.bin", float32_t)
READMAT(tensor_onnxMatMul_205, 120000, "onnxMatMul_205.bin", float32_t)
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

void run_inference(float32_t* tensor_onnxMatMul_0, float32_t* tensor_hidden1, float32_t* tensor_hidden, float32_t* tensor_163, float32_t* tensor_84, float32_t* tensor_148) {
omp_set_num_threads(4);

#ifdef COMPILER_BENCHMARK
double neuralcasting_time_benchmark = 0.0f;
double neuralcasting_end_benchmark = 0.0f;
double neuralcasting_start_benchmark = 0.0f;
#endif




// MATMUL OPERATOR fc1MatMul

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc1MatMul_output_0[1 * 200];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

#pragma omp parallel for shared(tensor_onnxMatMul_0, tensor_onnxMatMul_164, tensor_fc1MatMul_output_0) collapse(2)
for(int32_t i=0; i<1; i++) {
    for(int32_t j=0; j<200; j++) {
        float32_t temp = 0.0f;
#pragma omp reduction(temp: +)
        for(int32_t k=0; k<257; k++) {
            int32_t index1 = i*257+k;
            int32_t index2 = k*200+j;
            temp += tensor_onnxMatMul_0[index1] * tensor_onnxMatMul_164[index2];
        }
        tensor_fc1MatMul_output_0[i*200 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_fc1MatMul", 714)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc1MatMul -----------------\n");

for(int i=0; i<1; i++) {
    for(int j=0; j<200; j++) {
        printf("%f, ", tensor_fc1MatMul_output_0[i*200+j]);
    }
    printf("\n");
}

printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// ELEMENT WISE ADDITION fc1Add

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float32_t tensor_fc1Add_output_0[200];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
#pragma omp parallel for shared(tensor_fc1bias, tensor_fc1MatMul_output_0, tensor_fc1Add_output_0) private(i0, i1)
for(int i2=0; i2<200; i2++) {

tensor_fc1Add_output_0[i1*200 + i2*1] = tensor_fc1bias[i2*1] + tensor_fc1MatMul_output_0[i1*200 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_fc1Add", 400)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT fc1Add -----------------\n");
for(int i=0; i<200; i++) {
    printf("%f ", tensor_fc1Add_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TRANSPOSE /gru1/Transpose

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru1Transpose_output_0[200];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
for(int i2=0; i2<200; i2++) {

tensor_gru1Transpose_output_0[i0*(1*200)+i2*(1)] = tensor_fc1Add_output_0[i1*200 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru1/Transpose", 0)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/Transpose -----------------\n");
for(int i=0; i<200; i++) {
    printf("%f ", tensor_gru1Transpose_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// GRU OPERATOR /gru1/GRU

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru1GRU_output_0[200];
#undef CONNECTED_OUTPUT
#endif



#ifdef CONNECTED_HIDDEN
float tensor_84[200];
#undef CONNECTED_HIDDEN
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

{
    // a = W_ir @ x + b_ir
    float a[200];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        a[i] = 0.0f;
#pragma omp reduction(a: +)
        for(int j=0; j<200; j++) {
            a[i] += tensor_onnxGRU_182[(200*200) + i*200 + j] * tensor_gru1Transpose_output_0[j];
        }
        a[i] += tensor_onnxGRU_184[200 + i];
    }

    // b = W_hr @ h1 + b_hr
    float b[200];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        b[i] = 0.0f;
#pragma omp reduction(b: +)
        for(int j=0; j<200; j++) {
            b[i] += tensor_onnxGRU_183[(200*200) + i*200 + j] * tensor_hidden1[j];
        }
        b[i] += tensor_onnxGRU_184[4*200 + i];
    }

    // c = W_iz @ x + b_iz
    float c[200];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        c[i] = 0.0f;
#pragma omp reduction(c: +)
        for(int j=0; j<200; j++) {
            c[i] += tensor_onnxGRU_182[i*200 + j] * tensor_gru1Transpose_output_0[j];
        }
        c[i] += tensor_onnxGRU_184[i];
    }

    // d = W_hz @ h1 + b_hz
    float d[200];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        d[i] = 0.0f;
#pragma omp reduction(d: +)
        for(int j=0; j<200; j++) {
            d[i] += tensor_onnxGRU_183[i*200 + j] * tensor_hidden1[j];
        }
        d[i] += tensor_onnxGRU_184[3*200 + i];
    }

    // e = W_in @ x + b_in
    float e[200];
#pragma omp parallel for shared(tensor_onnxGRU_182, tensor_gru1Transpose_output_0,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        e[i] = 0.0f;
#pragma omp reduction(e: +)
        for(int j=0; j<200; j++) {
            e[i] += tensor_onnxGRU_182[(2*200*200) + i*200 + j] * tensor_gru1Transpose_output_0[j];
        }
        e[i] += tensor_onnxGRU_184[2*200 + i];
    }

    // f = W_hn @ h1 + b_hn
    float f[200];
#pragma omp parallel for shared(tensor_onnxGRU_183, tensor_hidden1,  tensor_onnxGRU_184) collapse(1)
    for(int i=0; i< 200; i++) {
        f[i] = 0.0f;
#pragma omp reduction(f: +)
        for(int j=0; j<200; j++) {
            f[i] += tensor_onnxGRU_183[(2*200*200) + i*200 + j] * tensor_hidden1[j];
        }
        f[i] += tensor_onnxGRU_184[5*200 + i];
    }

    // r = sigmoid(a + b)
    float r[200];
#pragma omp parallel for shared(a, b, r)
    for(int i=0; i<200; i++) {
        float s = a[i] + b[i];
        r[i] = 1.0f / (1.0f + expf(-s));
    }

    // z = sigmoid(c + d)
    float z[200];
#pragma omp parallel for shared(c, d, z)
    for(int i=0; i<200; i++) {
        float s = c[i] + d[i];
        z[i] = 1.0f / (1.0f + expf(-s));
    }

    // n = tanh(e + r*f)
    float n[200];
#pragma omp parallel for shared(n, e, r, f)
    for(int i=0; i<200; i++) {
        n[i] = tanh(e[i] + r[i] * f[i]);
    }

    // hn = (1-z) * n + z * h1
    float hn[200];
#pragma omp parallel for shared(tensor_84, z, n, tensor_hidden1, tensor_gru1GRU_output_0)
    for(int i=0; i<200; i++) {
        tensor_84[i] = (1.0f - z[i]) * n[i] + z[i] * tensor_hidden1[i];
        tensor_gru1GRU_output_0[i] = tensor_84[i];
    }
}

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru1/GRU", 485600)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/GRU -----------------\n");

for(int i=0; i<200; i++) {
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
for(int i=0; i<200; i++) {
    tensor_gru1Squeeze_output_0[i] = tensor_gru1GRU_output_0[i];
}
#endif

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru1/Squeeze", 0)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru1/Squeeze -----------------\n");
for(int i=0; i<200; i++) {
    printf("%f ", tensor_gru1Squeeze_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif




// GRU OPERATOR /gru2/GRU

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru2GRU_output_0[200];
#undef CONNECTED_OUTPUT
#endif



#ifdef CONNECTED_HIDDEN
float tensor_148[200];
#undef CONNECTED_HIDDEN
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

{
    // a = W_ir @ x + b_ir
    float a[200];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        a[i] = 0.0f;
#pragma omp reduction(a: +)
        for(int j=0; j<200; j++) {
            a[i] += tensor_onnxGRU_202[(200*200) + i*200 + j] * tensor_gru1Squeeze_output_0[j];
        }
        a[i] += tensor_onnxGRU_204[200 + i];
    }

    // b = W_hr @ h1 + b_hr
    float b[200];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        b[i] = 0.0f;
#pragma omp reduction(b: +)
        for(int j=0; j<200; j++) {
            b[i] += tensor_onnxGRU_203[(200*200) + i*200 + j] * tensor_hidden[j];
        }
        b[i] += tensor_onnxGRU_204[4*200 + i];
    }

    // c = W_iz @ x + b_iz
    float c[200];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        c[i] = 0.0f;
#pragma omp reduction(c: +)
        for(int j=0; j<200; j++) {
            c[i] += tensor_onnxGRU_202[i*200 + j] * tensor_gru1Squeeze_output_0[j];
        }
        c[i] += tensor_onnxGRU_204[i];
    }

    // d = W_hz @ h1 + b_hz
    float d[200];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        d[i] = 0.0f;
#pragma omp reduction(d: +)
        for(int j=0; j<200; j++) {
            d[i] += tensor_onnxGRU_203[i*200 + j] * tensor_hidden[j];
        }
        d[i] += tensor_onnxGRU_204[3*200 + i];
    }

    // e = W_in @ x + b_in
    float e[200];
#pragma omp parallel for shared(tensor_onnxGRU_202, tensor_gru1Squeeze_output_0,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        e[i] = 0.0f;
#pragma omp reduction(e: +)
        for(int j=0; j<200; j++) {
            e[i] += tensor_onnxGRU_202[(2*200*200) + i*200 + j] * tensor_gru1Squeeze_output_0[j];
        }
        e[i] += tensor_onnxGRU_204[2*200 + i];
    }

    // f = W_hn @ h1 + b_hn
    float f[200];
#pragma omp parallel for shared(tensor_onnxGRU_203, tensor_hidden,  tensor_onnxGRU_204) collapse(1)
    for(int i=0; i< 200; i++) {
        f[i] = 0.0f;
#pragma omp reduction(f: +)
        for(int j=0; j<200; j++) {
            f[i] += tensor_onnxGRU_203[(2*200*200) + i*200 + j] * tensor_hidden[j];
        }
        f[i] += tensor_onnxGRU_204[5*200 + i];
    }

    // r = sigmoid(a + b)
    float r[200];
#pragma omp parallel for shared(a, b, r)
    for(int i=0; i<200; i++) {
        float s = a[i] + b[i];
        r[i] = 1.0f / (1.0f + expf(-s));
    }

    // z = sigmoid(c + d)
    float z[200];
#pragma omp parallel for shared(c, d, z)
    for(int i=0; i<200; i++) {
        float s = c[i] + d[i];
        z[i] = 1.0f / (1.0f + expf(-s));
    }

    // n = tanh(e + r*f)
    float n[200];
#pragma omp parallel for shared(n, e, r, f)
    for(int i=0; i<200; i++) {
        n[i] = tanh(e[i] + r[i] * f[i]);
    }

    // hn = (1-z) * n + z * h1
    float hn[200];
#pragma omp parallel for shared(tensor_148, z, n, tensor_hidden, tensor_gru2GRU_output_0)
    for(int i=0; i<200; i++) {
        tensor_148[i] = (1.0f - z[i]) * n[i] + z[i] * tensor_hidden[i];
        tensor_gru2GRU_output_0[i] = tensor_148[i];
    }
}

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru2/GRU", 485600)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/GRU -----------------\n");

for(int i=0; i<200; i++) {
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
for(int i=0; i<200; i++) {
    tensor_gru2Squeeze_output_0[i] = tensor_gru2GRU_output_0[i];
}
#endif

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru2/Squeeze", 0)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/Squeeze -----------------\n");
for(int i=0; i<200; i++) {
    printf("%f ", tensor_gru2Squeeze_output_0[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
// TRANSPOSE /gru2/Transpose

#define CONNECTED_OUTPUT

#ifdef CONNECTED_OUTPUT
float tensor_gru2Transpose_output_0[200];
#undef CONNECTED_OUTPUT
#endif

#ifdef COMPILER_BENCHMARK
neuralcasting_start_benchmark = omp_get_wtime();
#endif

for(int i0=0; i0<1; i0++) {
for(int i1=0; i1<1; i1++) {
for(int i2=0; i2<200; i2++) {

tensor_gru2Transpose_output_0[i0*(1*200)+i2*(1)] = tensor_gru2Squeeze_output_0[i1*200 + i2*1];
}
}
}


#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_/gru2/Transpose", 0)
#endif

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT /gru2/Transpose -----------------\n");
for(int i=0; i<200; i++) {
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
        for(int32_t k=0; k<200; k++) {
            int32_t index1 = i*200+k;
            int32_t index2 = k*600+j;
            temp += tensor_gru2Transpose_output_0[index1] * tensor_onnxMatMul_205[index2];
        }
        tensor_fc3MatMul_output_0[i*600 + j] = temp;
    }
}

#ifdef COMPILER_BENCHMARK
BENCHMARK("tensor_fc3MatMul", 1000)
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
BENCHMARK("tensor_fc3Add", 1200)
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
BENCHMARK("tensor_reluRelu", 0)
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
BENCHMARK("tensor_fc4MatMul", 1800)
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
BENCHMARK("tensor_fc4Add", 1200)
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
BENCHMARK("tensor_relu_1Relu", 0)
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
BENCHMARK("tensor_fc2MatMul", 1457)
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
BENCHMARK("tensor_fc2Add", 514)
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
BENCHMARK("tensor_sigmoidSigmoid", 1542)
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