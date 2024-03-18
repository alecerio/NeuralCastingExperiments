#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <fstream>

#define NUM_EXPERIMENTS (10000)

void write_to_file(const std::string& filename, const double* data, size_t size);

int main(int argc, char* argv[]) {

    const char* model_path = argv[1]; 
    const int SIZE_INPUT = std::atoi(argv[2]);
    const int SIZE_HIDDEN = std::atoi(argv[3]);
    const char* output_path = argv[4];

    // create onnxruntime env
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // create ort session
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    // prepare input data
    const int input_tensor_size = SIZE_INPUT;
    float input_data[input_tensor_size];
    for(int i=0; i<input_tensor_size; i++)
        input_data[i] = 1.0f;
    
    // prepare hidden state 1
    const int hidden_tensor_size = SIZE_HIDDEN;
    float hidden_data_1[hidden_tensor_size];
    for(int i=0; i<hidden_tensor_size; i++)
        hidden_data_1[i] = 0.0f;

    // prepare hidden state 2
    float hidden_data_2[hidden_tensor_size];
    for(int i=0; i<hidden_tensor_size; i++)
        hidden_data_2[i] = 0.0f;
    
    // Define OrtMemoryInfo for CPU memory
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // Create Ort::Value with the input tensor
    int64_t* input_shape = (int64_t*) malloc(sizeof(int64_t)*3);
    input_shape[0] = 1; input_shape[1] = 1; input_shape[2] = SIZE_INPUT;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_tensor_size, input_shape, 3);

    // Create Ort::Value with the hidden tensor
    int64_t* hidden_shape = (int64_t*) malloc(sizeof(int64_t)*3);
    hidden_shape[0] = 1; hidden_shape[1] = 1; hidden_shape[2] = SIZE_HIDDEN;
    
    Ort::Value hidden_tensor_1 = Ort::Value::CreateTensor<float>(memory_info, hidden_data_1, hidden_tensor_size, hidden_shape, 3);
    Ort::Value hidden_tensor_2 = Ort::Value::CreateTensor<float>(memory_info, hidden_data_2, hidden_tensor_size, hidden_shape, 3);
    
    // run inference
    const char* input_names[] = {"onnx::MatMul_0", "hidden.1", "hidden"};
    const char* output_names[] = {"81", "141", "155"};

    double* times = new double[NUM_EXPERIMENTS];
    double start_time, end_time, time;
    for(int i=0; i<NUM_EXPERIMENTS; i++) {
        // run inference
        start_time = omp_get_wtime();
        std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(nullptr), input_names, &input_tensor, 3, output_names, 3);
        end_time = omp_get_wtime();
        time = end_time - start_time;
        
        // updated times
        times[i] = time;
    }

    write_to_file(output_path, times, NUM_EXPERIMENTS);

    delete[] times;

    return 0;
}

void write_to_file(const std::string& filename, const double* data, size_t size) {
    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        for (size_t i = 0; i < size; ++i) {
            outfile << data[i];
            if (i != size - 1) {
                outfile << ";";
            }
        }
        outfile << std::endl;
        outfile.close();
    } else {
        std::cerr << "Error opening the file " << filename << "." << std::endl;
    }
}