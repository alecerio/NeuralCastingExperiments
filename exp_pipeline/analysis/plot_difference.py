import numpy as np
import matplotlib.pyplot as plt

input_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
hidden_size : list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

nc_avgmat_path = 'avgmat_nc.txt'
ort_avgmat_path = 'avgmat_ort.txt'

nc_avgmat = np.loadtxt(nc_avgmat_path)
ort_avgmat = np.loadtxt(ort_avgmat_path)

difference_matrix = nc_avgmat - ort_avgmat

plt.imshow(difference_matrix, cmap='coolwarm', interpolation='bicubic', extent=[50, 1000, 1000, 50], vmin=-0.002, vmax=0.002)
#plt.colorbar()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.title('average latency difference [s]', fontsize=20)
plt.xlabel('input size', fontsize=16)
plt.ylabel('gru hidden size', fontsize=16)
plt.xticks(input_size, rotation=45, fontsize=16)
plt.yticks(hidden_size, fontsize=16) 
plt.show()

#plt.figure(figsize=(8, 6))
#plt.imshow(difference_matrix, cmap='coolwarm', interpolation='bicubic', vmin=-0.001, vmax=0.001)
#plt.colorbar(label='Difference')
#plt.title('Difference Between Matrices 1 and 2')
#plt.xlabel('X Axis')
#plt.ylabel('Y Axis')
#plt.show()
