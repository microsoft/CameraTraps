##############################################################
# ct_utils.py
#
# Script with shared utility functions, such as truncating floats
##############################################################
import math
import numpy as np

def truncate_float_array(xs, precision=3):
    ''' Vectorized version of truncate_float(...) '''
    vectorized_fun = np.vectorize(lambda x: truncate_float(x, precision))
    return vectorized_fun(xs).tolist()

def truncate_float(x, precision=3):
    ''' 
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON

    Args: 
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserver, should be 
                      greater or equal 1
    '''
    assert precision > 0
    
    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10,precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and 
        # division by factor
        return math.floor(x * factor)/factor