import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(in_features=257, out_features=400, bias=True)  # Set bias=False to separate MatMul and Add
        self.fc3 = nn.Linear(in_features=400, out_features=600, bias=True)  # Set bias=False to separate MatMul and Add
        self.fc4 = nn.Linear(in_features=600, out_features=600, bias=True)  # Set bias=False to separate MatMul and Add
        self.fc2 = nn.Linear(in_features=600, out_features=257, bias=True)  # Set bias=False to separate MatMul and Add

        self.gru1 = nn.GRU(input_size=400, hidden_size=400, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=400, hidden_size=400, num_layers=1, batch_first=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h1=None, h2=None):
        # fc1
        x = self.fc1(x)
        #x = self.relu(x)

        # gru1
        x, h1 = self.gru1(x, h1)

        # gru2
        x, h2 = self.gru2(x, h2)

        # fc3
        x = self.fc3(x)
        x = self.relu(x)

        # fc4
        x = self.fc4(x)
        x = self.relu(x)

        # fc2
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x, h1, h2

# Instantiate the model
model = Model()

# Print the model architecture
print(model)

import torch.onnx

# Instantiate the model
model = Model()

# Create dummy input tensor
dummy_input = torch.randn(1, 1, 257)

# Create initial hidden states
h1 = torch.zeros(1, 1, 400)
h2 = torch.zeros(1, 1, 400)

# Export the model to ONNX format
onnx_path = "model2.onnx"
torch.onnx.export(model, (dummy_input, h1, h2), onnx_path, opset_version=14)

print(f"Model exported to {onnx_path}")
