import matplotlib.pyplot as plt
import math
import numpy as np

def read_numbers_from_file(filename):
    numbers = []
    with open(filename, 'r') as file:
        for line in file:
            numbers.extend(map(float, line.strip().split(';')))
    return numbers

def plot_data(label, fc1matmul, fc1add, gru1transpose, gru1gru, gru1squeeze, gru2gru, gru2squeeze, gru2transpose, fc3matmul, fc3add, relurelu, fc4matmul, fc4add, relu1relu, fc2matmul, fc2add, sigmoidsigmoid):
    num_exp = len(fc1matmul)
    if label == 'avg':
        fc1matmul_avg = sum(fc1matmul) / num_exp
        fc1add_avg = sum(fc1add) / num_exp
        gru1transpose_avg = sum(gru1transpose) / num_exp
        gru1gru_avg = sum(gru1gru) / num_exp
        gru1squeeze_avg = sum(gru1squeeze) / num_exp
        gru2gru_avg = sum(gru2gru) / num_exp
        gru2squeeze_avg = sum(gru2squeeze) / num_exp
        gru2transpose_avg = sum(gru2transpose) / num_exp
        fc3matmul_avg = sum(fc3matmul) / num_exp
        fc3add_avg = sum(fc3add) / num_exp
        relurelu_avg = sum(relurelu) / num_exp
        fc4matmul_avg = sum(fc4matmul) / num_exp
        fc4add_avg = sum(fc4add) / num_exp
        relu1relu_avg = sum(relu1relu) / num_exp
        fc2matmul_avg = sum(fc2matmul) / num_exp
        fc2add_avg = sum(fc2add) / num_exp
        sigmoidsigmoid_avg = sum(sigmoidsigmoid) / num_exp

        avgs = [fc1matmul_avg, fc1add_avg, gru1transpose_avg, gru1gru_avg, gru1squeeze_avg, gru2gru_avg, gru2squeeze_avg, gru2transpose_avg, fc3matmul_avg, fc3add_avg,
                relurelu_avg, fc4matmul_avg, fc4add_avg, relu1relu_avg, fc2matmul_avg, fc2add_avg, sigmoidsigmoid_avg]
        
        modules = ['fc1matmul', 'fc1add', 'gru1transpose', 'gru1gru', 'gru1squeeze', 'gru2gru', 'gru2squeeze', 'gru2transpose', 'fc3matmul', 'fc3add', 'relurelu', 'fc4matmul',
                   'fc4add', 'relu1relu', 'fc2matmul', 'fc2add', 'sigmoidsigmoid']
        
        plt.bar(modules, avgs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.xlabel('module')
        plt.ylabel('latency [s]')
        plt.title('average latency')
        plt.show()
    
    elif label == 'stddev':
        fc1matmul_std = np.std(fc1matmul)
        fc1add_std = np.std(fc1add)
        gru1transpose_std = np.std(gru1transpose)
        gru1gru_std = np.std(gru1gru)
        gru1squeeze_std = np.std(gru1squeeze)
        gru2gru_std = np.std(gru2gru)
        gru2squeeze_std = np.std(gru2squeeze)
        gru2transpose_std = np.std(gru2transpose)
        fc3matmul_std = np.std(fc3matmul)
        fc3add_std = np.std(fc3add)
        relurelu_std = np.std(relurelu)
        fc4matmul_std = np.std(fc4matmul)
        fc4add_std = np.std(fc4add)
        relu1relu_std = np.std(relu1relu)
        fc2matmul_std = np.std(fc2matmul)
        fc2add_std = np.std(fc2add)
        sigmoidsigmoid_std = np.std(sigmoidsigmoid)

        stds = [fc1matmul_std, fc1add_std, gru1transpose_std, gru1gru_std, gru1squeeze_std, gru2gru_std, gru2squeeze_std, gru2transpose_std, fc3matmul_std, fc3add_std,
                relurelu_std, fc4matmul_std, fc4add_std, relu1relu_std, fc2matmul_std, fc2add_std, sigmoidsigmoid_std]
        
        modules = ['fc1matmul', 'fc1add', 'gru1transpose', 'gru1gru', 'gru1squeeze', 'gru2gru', 'gru2squeeze', 'gru2transpose', 'fc3matmul', 'fc3add', 'relurelu', 'fc4matmul',
                   'fc4add', 'relu1relu', 'fc2matmul', 'fc2add', 'sigmoidsigmoid']
        
        plt.bar(modules, stds)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.xlabel('module')
        plt.ylabel('latency [s]')
        plt.title('standard deviation latency')
        plt.show()



NUM_EXPERIMENTS = 10000

modules = {
    'FC1MATMUL': 0,
    'FC1ADD': 1,
    'GRU1TRANSPOSE': 2,
    'GRU1GRU': 3,
    'GRU1SQUEEZE': 4,
    'GRU2GRU': 5,
    'GRU2SQUEEZE': 6,
    'GRU2TRANSPOSE': 7,
    'FC3MATMUL': 8,
    'FC3ADD': 9,
    'RELURELU': 10,
    'FC4MATMUL': 11,
    'FC4ADD': 12,
    'RELU1RELU': 13,
    'FC2MATMUL': 14,
    'FC2ADD': 15,
    'SIGMOIDSIGMOID': 16
}

NUM_MODULES = len(modules.keys())

fc1matmul = []
fc1add = []
gru1transpose = []
gru1gru = []
gru1squeeze = []
gru2gru = []
gru2squeeze = []
gru2transpose = []
fc3matmul = []
fc3add = []
relurelu = []
fc4matmul = []
fc4add = []
relu1relu = []
fc2matmul = []
fc2add = []
sigmoidsigmoid = []

path = '/home/alessandro/Desktop/exp/NeuralCastingExperiments/nsnet_H400/'
filename = 'output.txt'
numbers = read_numbers_from_file(path+filename)

for i in range(0, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc1matmul.append(numbers[i])

for i in range(1, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc1add.append(numbers[i])

for i in range(2, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru1transpose.append(numbers[i])

for i in range(3, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru1gru.append(numbers[i])

for i in range(4, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru1squeeze.append(numbers[i])

for i in range(5, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru2gru.append(numbers[i])

for i in range(6, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru2squeeze.append(numbers[i])

for i in range(7, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    gru2transpose.append(numbers[i])

for i in range(8, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc3matmul.append(numbers[i])

for i in range(9, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc3add.append(numbers[i])

for i in range(10, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    relurelu.append(numbers[i])

for i in range(11, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc4matmul.append(numbers[i])

for i in range(12, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc4add.append(numbers[i])

for i in range(13, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    relu1relu.append(numbers[i])

for i in range(14, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc2matmul.append(numbers[i])

for i in range(15, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    fc2add.append(numbers[i])

for i in range(16, NUM_EXPERIMENTS*NUM_MODULES, NUM_MODULES):
    sigmoidsigmoid.append(numbers[i])

plot_data('avg', fc1matmul, fc1add, gru1transpose, gru1gru, gru1squeeze, gru2gru, gru2squeeze, gru2transpose, fc3matmul, fc3add, relurelu, fc4matmul, fc4add, relu1relu, fc2matmul, fc2add, sigmoidsigmoid)
