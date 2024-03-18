import torch
import torch.nn as nn
import torch.onnx
import sys

class Model(nn.Module):
    def __init__(self, input_size = 257, hidden_size=400):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)  
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=600, bias=True)  
        self.fc4 = nn.Linear(in_features=600, out_features=600, bias=True) 
        self.fc2 = nn.Linear(in_features=600, out_features=input_size, bias=True)

        self.gru1 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h1=None, h2=None):
        x = self.fc1(x)
        x, h1 = self.gru1(x, h1)
        x, h2 = self.gru2(x, h2)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, h1, h2

input_size = int(sys.argv[1])
hidden_size = int(sys.argv[2])
output_dir = str(sys.argv[3])

# Instantiate the model
model = Model(input_size, hidden_size)

# Create dummy input tensor
dummy_input = torch.randn(1, 1, input_size)

# Create initial hidden states
h1 = torch.zeros(1, 1, hidden_size)
h2 = torch.zeros(1, 1, hidden_size)

# Export the model to ONNX format
onnx_path = output_dir + '/nsnet_' + str(input_size) + '_' + str(hidden_size) + '.onnx'
torch.onnx.export(model, (dummy_input, h1, h2), onnx_path, opset_version=11)
