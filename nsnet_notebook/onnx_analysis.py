import onnx

# Load the ONNX model
model_path = "nsnet2.onnx"
onnx_model = onnx.load(model_path)

# Get the model graph
model_graph = onnx_model.graph

# Print basic information about the model
print("ONNX Model Information:")
print("Model Producer:", onnx_model.producer_name)
print("Model Version:", onnx.__version__)
print("IR Version:", onnx_model.ir_version)

# Print information about each node in the model graph
print("\nNodes in the ONNX model:")
for i, node in enumerate(model_graph.node):
    print(f"Node {i} - Name: {node.name}, OpType: {node.op_type}")
    print("  Inputs:")
    for input_name in node.input:
        print(f"    {input_name}")
    print("  Outputs:")
    for output_name in node.output:
        print(f"    {output_name}")

    # Print attributes of the node if available
    if node.attribute:
        print("  Attributes:")
        for attr in node.attribute:
            print(f"    {attr.name}: {attr}")
    print()
