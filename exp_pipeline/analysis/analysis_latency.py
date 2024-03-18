import os
import numpy as np
import statistics
import matplotlib.pyplot as plt

def compute_mean(values : list[float]) -> float:
    avg : float = statistics.mean(values)
    return avg

def compute_stddev(values : list[float]) -> float:
    std_dev : float = statistics.stdev(values)
    return std_dev

def compute_min(values : list[float]) -> float:
    min_val : float = min(values)
    return min_val

def compute_max(values : list[float]) -> float:
    max_val : float = max(values)
    return max_val

def read_results(input_size : int, hidden_size : int, results_path : str) -> list[float]:
    file_path : str = results_path + '/output_ort_' + str(input_size) + '_' + str(hidden_size) + '.txt'
    with open(file_path, 'r') as file:
        content = file.read()
    results : list[float] = [float(num) for num in content.split(';')]
    return results

# current directory
script_directory = os.path.dirname(os.path.abspath(__file__))
results_path = script_directory + '/../compile_nn_ort/results'

# define input and hidden size to analyse
input_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
hidden_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
n_input_size : int = len(input_size)
n_hidden_size : int = len(hidden_size)

# initialize result matrices (average, standard deviation, minimum, maximum)
avgmat = np.zeros((n_input_size, n_hidden_size), dtype=float)
stdmat = np.zeros((n_input_size, n_hidden_size), dtype=float)
minmat = np.zeros((n_input_size, n_hidden_size), dtype=float)
maxmat = np.zeros((n_input_size, n_hidden_size), dtype=float)

for i_idx in range(0, n_input_size):
    for h_idx in range(0, n_hidden_size):
        i : int = input_size[i_idx]
        h : int = hidden_size[h_idx]

        # retrive results
        res : list[float] = read_results(i, h, results_path)

        # compute average value
        avgval : float = compute_mean(res)
        avgmat[i_idx][h_idx] = avgval

        # compute standard deviation value
        stddevval : float = compute_stddev(res)
        stdmat[i_idx][h_idx] = stddevval

        # compute minimum value
        minval : float = compute_min(res)
        minmat[i_idx][h_idx] = minval

        # compute maximum value
        maxval : float = compute_max(res)
        maxmat[i_idx][h_idx] = maxval


np.savetxt('avgmat_ort.txt', avgmat, fmt='%f')

mat = stdmat
for inp_idx, inp in enumerate(input_size):
    for hid_idx, hid in enumerate(hidden_size):
        if inp % 100 == 0 and hid % 100 == 0:
            print(str(inp) + " - " + str(hid) + " - " + str(mat[inp_idx][hid_idx]))

# plot heat map average
plt.imshow(avgmat, cmap='hot', interpolation='bicubic', extent=[50, 1000, 1000, 50])
#plt.colorbar()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.title('average latency [s]', fontsize=20)
plt.xlabel('input size', fontsize=16)
plt.ylabel('gru hidden size', fontsize=16)
plt.xticks(input_size, rotation=45, fontsize=16)
plt.yticks(hidden_size, fontsize=16) 
plt.show()