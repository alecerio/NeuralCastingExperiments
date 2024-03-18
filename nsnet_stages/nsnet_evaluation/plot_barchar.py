import matplotlib.pyplot as plt
import numpy as np

#modules = modules = ['fc1matmul', 'fc1add', 'gru1transpose', 'gru1gru', 'gru1squeeze', 
#                     'gru2gru', 'gru2squeeze', 'gru2transpose', 'fc3matmul', 'fc3add', 'relurelu', 'fc4matmul',
#                   'fc4add', 'relu1relu', 'fc2matmul', 'fc2add', 'sigmoidsigmoid']

modules = modules = ['matmul', 'add', 'transpose', 'gru', 'squeeze', 
                     'gru', 'squeeze', 'transpose', 'matmul', 'add', 'relu', 'matmul',
                   'add', 'relu', 'matmul', 'add', 'sigmoid']

avg_1000_400 = [0.00011664227497000035, 9.215986700000046e-07, 1.2373191999999858e-07, 0.00028785447513000065, 
               2.4058199999998435e-08, 0.00028793478065999997, 2.4037049999998483e-08, 1.3847482999999698e-07, 
               7.640801495000013e-05, 9.398461000000073e-07, 9.092007000000032e-07, 0.0001148249365100004, 
               9.620850700000052e-07, 8.994696700000008e-07, 0.0001850379953299995, 9.98200960000006e-07, 2.421577000000002e-06]

avg_600_400 = [6.675187455000004e-05, 8.568252600000029e-07, 1.2372840999999813e-07, 0.0002782249014799997, 2.3250319999997518e-08, 
               0.00027623722470999945, 2.3289879999997496e-08, 1.6039041000000138e-07, 7.19794280799998e-05, 8.642319300000056e-07, 
               8.332792200000024e-07, 0.00010959118575000018, 8.706457100000021e-07, 8.124652100000053e-07, 0.00010692208754000062, 
               8.573599100000007e-07, 1.690761729999998e-06]

avg_100_400 = [8.446571509999993e-06, 8.002026100000115e-07, 1.0564990000000065e-07, 0.00024770527226000005, 2.1063139999998494e-08, 
               0.0002511019887100008, 2.154740999999829e-08, 1.1754978000000081e-07, 6.0191632920000095e-05, 8.774238300000004e-07, 
               8.268411000000036e-07, 9.102624639000002e-05, 8.772592000000039e-07, 8.302629400000058e-07, 1.604406388000007e-05, 
               8.240836600000074e-07, 8.908401900000074e-07]


bar_width = 0.2
x = np.arange(len(modules))
plt.bar(x - bar_width, avg_100_400, width=bar_width, label='IS=100, HS=400')
plt.bar(x, avg_600_400, width=bar_width, label='IS=600, HS=400')
plt.bar(x + bar_width, avg_1000_400, width=bar_width, label='IS=1000, HS=400')

plt.xlabel('Modules', fontsize=20)
plt.ylabel('Latency Time [s]', fontsize=20)
plt.title('Latency Time for Different Input Sizes', fontsize=24)
plt.xticks(x, modules, fontsize=18, rotation=45)
plt.yticks(fontsize=20)
plt.legend(fontsize='large')
plt.show()