"""File containing definitions for the controllers"""
import cvxpy as cp

from .typedefs import *
from numba import jit
from scipy.signal import place_poles

# FIXME: broken due to using complex datatype
def create_linear_continuous_time_controller(A: Mat, B: Mat, poles) -> fn[[Mat], Mat]:
    K: Mat = place_poles(A, B, poles).gain_matrix
    @jit(nopython=True)
    def c(x: Mat) -> Mat: return -K@x
    return c

