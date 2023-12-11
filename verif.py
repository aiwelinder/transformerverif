import numpy as np
from z3 import *
import onnx
from onnx import numpy_helper
import torchvision.datasets as datasets

EPSILON_VALS = [0.016, 0.02, 0.024, 0.032]

# NOTE: this code is hard-coded to work on the self-attention-mnist-small.onnx file
TEST_NETWORK = './training/self-attention-mnist-small.onnx'
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

def toy_example_two():
    # basic ass softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Example input to softmax
    input_scores = np.array([12, 15, 18])

    # Calculate softmax
    softmax_output = softmax(input_scores)

    # create z3 solver and add convert np arr to z3
    solver = Solver()
    softmax_vars = [RealVal(val) for val in softmax_output]

    lower_bound = RealVal(0.1)
    upper_bound = RealVal(0.9)

    # Add constriants to the solver for each softmax variable and check
    for var in softmax_vars:
        solver.add(var >= lower_bound, var <= upper_bound)

    if solver.check() == sat:
        print("Solution exists within the bounds.")
        print("One such solution: ", solver.model())
    else:
        print("No solution exists within the bounds.")

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
            # This runs until we have saved the 
            print(previous_previous_layer.op_type)
            print(previous_layer.op_type)
            break
        previous_previous_layer = previous_layer
        previous_layer = layer
    
    # Now, we have isolated the MatMul (weights), Add (bias), and the Softmax layers
    weights = extract_weights(previous_previous_layer)
    print("Hey it's the weights! ", weights)
    biases = extract_biases(previous_layer)
    print("Hey it's the biases! ", biases)

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    X_test = mnist_testset.data.numpy()
    Y_test = mnist_testset.targets.numpy()
    point = np.array(X_test[0]).flatten() / 255
    networkOutput = compute_softmax_values(weights, biases, point)
    expected_output = None


    # TODO: We need a 16x10 vector of inputs???? In addition to their expected output???
    input_val = compute_softmax_inputs(weights, biases, )

    ## DEAD CODE BELOW 
    # input_val = Symbol("input", BV32)

    # # add actual softmax logic shit here but for now
    # remainder = BVURem(input_val, BV(10, 32))
    
    # leq_5 = And(BVUGE(remainder, BV(0, 32)), BVULE(remainder, BV(5, 32)))
    # # greater_than_5 = And(GE(remainder, BV(6, 32)), LE(remainder, BV(9, 32)))
    # nearest_odd_multiple = Ite(leq_5, BVAdd(input_val, BVSub(BV(5, 32), remainder)), BVSub(input_val, BVSub(BV(5, 32), remainder)))

    # actual_output = BV(25, 32)
    # operations_prop = Equals(actual_output, nearest_odd_multiple)
    # range_prop = And(BVULE(input_val, BVAdd(actual_output, BV(4, 32))), BVUGE(input_val, BVSub(actual_output, BV(5, 32))))
    
    # solver = Solver()
    # solver.add_assertion(Or(Not(operations_prop), Not(range_prop)))
    # solver.solve()

    # if not solver.solve():
    #     print("Property holds for the following model:")
    #     print(solver.get_model())
    # else:
    #     print("Property does not hold.")

    # """
    # actual transformer
    
    # define input matrix as symbol
    # we know what the output should be, store in "actual_output" var or smth

    # do all the operations
    # property to verify:
    #     result of operations is NOT equal to actual_output
    #     the input is NOT within the range
    #     take disjunction of above, make sure it is unsat

    # ask: logic (does the disjunction of negations work? right now it's saying property does
    # not hold for our trivial example)
    # ask: implementation of actual softmax
    #     we know we should initialize a matrix, but after that. crickets
    #     can we just import onnx and like. run it

    #      - implement and encode softmax function using exponential encoding pySMT 
    #      - then it's a question  of whether we want to add another encoding via log exponential (in paper) or some improvement

    #      - if we were to encode sigmoid using bit vector
         
    # ask: is there more than one operation bc it looks like a billion
    
    # """

if __name__ == '__main__':
    network()
    toy_example()
    toy_example_two()
    
