
import math
def exponential_decay(init_value, time_decay):
    '''
    returns a function with exponential decay from the init_value
    as exposed in the next equation:
    f(epoch): init_value * exp(-epoch/delta)
    Params:
    - init_value (float): the initial radius
    - time_decay (float): the rate of decay (should cosider > 1 as lower decay)

    Return:
    function signature: (epoch) -> float
    '''

    return lambda epoch: init_value * math.exp(- epoch / time_decay)


def linear_decay(init_val, max_epochs):
    '''
    Returns a function with linear decay from the init_value
    as exposed in the next equation:
    f(epoch): (((max_epochs - epoch * 1.2) * init_val) / max_epochs) + 1
    Params:
    - init_value (float): the initial radius
    - time_decay (float): the rate of decay (should cosider > 1 as lower decay)

    Return:
    function signature: (epoch) -> float
    '''
    return lambda epoch: (((max_epochs - epoch * 1.2) * init_val) / (max_epochs + 1))
