import numpy as np
from z3 import *
import onnx
from onnx import numpy_helper
import idx2numpy
import onnxruntime as ort

EPSILON_VALS = [0.016, 0.02, 0.024, 0.032]

# NOTE: this code is hard-coded to work on the self-attention-mnist-small.onnx file
TEST_NETWORK = './training/self-attention-mnist-small.onnx'
PRUNED_NETWORK = './modified_pruned_network.onnx'
onnx_model = onnx.load(TEST_NETWORK)

# Toy model for how our code with the REAL softmax should generally be architect4ed
def toy_example():
    # Instead this should be the softmax fn
    def f(x):
        return x**2

    # Create a Z3 Solver
    solver = Solver()

    # Define a Real variable
    x = Real('x')

    # Define the bounds
    lower_bound = RealVal(10)
    upper_bound = RealVal(20)

    # Add constraints to the solver
    solver.add(f(x) >= lower_bound, f(x) <= upper_bound)

    # Check if the constraints are satisfiable
    if solver.check() == sat:
        print("Solution exists within the bounds.")
        print("One such solution: ", solver.model())
    else:
        print("No solution exists within the bounds.")

# basic ass softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_softmax_values(weights, biases, input_data):
    logits = np.dot(input_data, weights) + biases
    probabilities = softmax(logits)
    return probabilities

def extract_weights(layer):
    # Parse the onnx_model.graph.initializer list, which contains the model's parameters
    for tensor in onnx_model.graph.initializer:
        if tensor.name == layer.input[1]:
            # Convert the ONNX tensor to a numpy array
            return numpy_helper.to_array(tensor)
    raise ValueError("Weights not found for layer")

def extract_biases(layer):
    # Extract biases, which may be the second input to the Add layer
    for tensor in onnx_model.graph.initializer:
        if tensor.name == layer.input[1]:
            return numpy_helper.to_array(tensor)
    raise ValueError("Biases not found for layer")

def network():
    # In order to manually acquire the weights and biases for this particular network, we need to traverse the
    # network until we find the softmax layer, and store the previous two layers, which contain the weights and bias
    previous_previous_layer = None
    previous_layer = None
    for layer in onnx_model.graph.node:
        if layer.op_type == 'Softmax' and previous_previous_layer.op_type == 'MatMul':
            # This runs until we have saved the 2 layers preceding the softmax step
            # print(previous_previous_layer.op_type)
            # print(previous_layer.op_type)
            break
        previous_previous_layer = previous_layer
        previous_layer = layer
    
    # Now, we have isolated the MatMul (weights), Add (bias), and the Softmax layers
    weights = extract_weights(previous_previous_layer)
    # print("Hey it's the weights! ", weights)
    biases = extract_biases(previous_layer)
    # print("Hey it's the biases! ", biases)

    # Previously, we tried using the torchvision dataset but instead downloaded it directly and threw it into the repo
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    # X_test = mnist_testset.data.numpy()
    # Y_test = mnist_testset.targets.numpy()
    images_file = 'archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    labels_file = 'archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    # Load the images and labels from IDX format files
    X_test = idx2numpy.convert_from_file(images_file)
    Y_test = idx2numpy.convert_from_file(labels_file)
    # Load the ONNX model using ONNX Runtime
    # This network's final 3 layers are pruned s.t. its output is the input to the softmax fn.
    sess = ort.InferenceSession(PRUNED_NETWORK)

    # Reshape the MNIST image data to the expected input shape of the model
    input_details = sess.get_inputs()[0]
    input_name = input_details.name
    expected_input_shape = input_details.shape
    # print("Expected input shape:", expected_input_shape)
    input_data = X_test.astype('float32') / 255
    input_data = np.reshape(input_data, (input_data.shape[0], 28, 28, 1))  # Change to NHWC format
    
    # Run the model up to output name of the Reshape layer right before the MatMul leading to softmax
    reshape_output_name = sess.get_outputs()[0].name
    outputs = sess.run([reshape_output_name] ,{input_name: input_data})

    # 'outputs' now contains the output of the layer before the MatMul that feeds into the softmax
    input_to_matmul = outputs[0]

    # Reshape input_to_matmul to match expected weights and compute softmax
    input_to_matmul = input_to_matmul.reshape(input_to_matmul.shape[0], -1)
    networkOutput = compute_softmax_values(weights, biases, input_to_matmul)
    # Convert weights into labels
    class_labels = np.argmax(networkOutput, axis=1)

    # Now we do formal methods!!
    solver = Solver()
    output_to_check = [RealVal(val) for val in class_labels]
    expected_output = [RealVal(val) for val in Y_test]

    for eps in EPSILON_VALS:
        print("Running check for range: ", eps)
        for i in range(len(expected_output)):
            solver.add(output_to_check[i] >= RealVal(expected_output[i] - eps))
            solver.add(output_to_check[i] <= RealVal(expected_output[i] + eps))

        if solver.check() == sat:
            print("Solution exists within the bounds.")
            print("One such solution: ", solver.model())
        else:
            print("No solution exists within the bounds.")


if __name__ == '__main__':
    toy_example()
    network()
