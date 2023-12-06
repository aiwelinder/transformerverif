from pysmt.shortcuts import *
from pysmt.typing import *

# create a type of Bit-Vector of size 32.
BV32 = BVType(32)

### Network architectur
def network():
    input_val = Symbol("input", BV32)

    # add actual softmax logic shit here but for now
    remainder = BVURem(input_val, BV(10, 32))
    
    leq_5 = And(BVUGE(remainder, BV(0, 32)), BVULE(remainder, BV(5, 32)))
    # greater_than_5 = And(GE(remainder, BV(6, 32)), LE(remainder, BV(9, 32)))
    nearest_odd_multiple = Ite(leq_5, BVAdd(input_val, BVSub(BV(5, 32), remainder)), BVSub(input_val, BVSub(BV(5, 32), remainder)))

    actual_output = BV(25, 32)
    operations_prop = Equals(actual_output, nearest_odd_multiple)
    range_prop = And(BVULE(input_val, BVAdd(actual_output, BV(4, 32))), BVUGE(input_val, BVSub(actual_output, BV(5, 32))))
    
    solver = Solver()
    solver.add_assertion(Or(Not(operations_prop), Not(range_prop)))
    solver.solve()


    """
    actual transformer
    
    define input matrix as symbol
    we know what the output should be, store in "actual_output" var or smth

    do all the operations
    property to verify:
        result of operations is NOT equal to actual_output
        the input is NOT within the range
        take disjunction of above, make sure it is unsat

    ask: logic (does the disjunction of negations work? right now it's saying property does
    not hold for our trivial example)
    ask: implementation of actual softmax
        we know we should initialize a matrix, but after that. crickets
        can we just import onnx and like. run it
    ask: is there more than one operation bc it looks like a billion
    
    """
    if not solver.solve():
        print("Property holds for the following model:")
        print(solver.get_model())
    else:
        print("Property does not hold.")

if __name__ == '__main__':
    network()
    