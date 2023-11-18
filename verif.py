from pysmt.shortcuts import *
from pysmt.typing import *

def symbol_at_time(sym, k):
    return Symbol('{}@{}'.format(sym.symbol_name(), k), sym.symbol_type())

### Bounded Model Checking loop
def bmc(max_bound):
    return

if __name__ == '__main__':
    bmc(10)
    
