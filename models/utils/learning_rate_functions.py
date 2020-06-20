
import math

def constant(init_val):
    '''
    Returns a function which only returns the same init_value
    '''
    return lambda x : init_val


def exponential_decay(init_val, time_decay):
    '''
    returns a function with exponential decay from the init_value
    as exposed in the next equation:
    f(epoch): init_value * exp(-epoch/delta)
    Params:
    - init_value (float): the initial learning reate
    - time_decay (float): the rate of decay (should cosider > 1 as lower decay)

    Return:
    function signature: (epoch) -> float
    '''

    return lambda epoch: init_val * math.exp(- epoch / time_decay)

