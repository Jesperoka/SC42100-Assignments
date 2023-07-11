"""File containing definitions for the controllers"""
import cvxpy as cp
import numpy as np

#from typedefs import *
from .typedefs import *
from numba import jit
from scipy.signal import place_poles

# FIXME: broken due to using complex datatype
def create_linear_continuous_time_controller(A: Mat, B: Mat, poles) -> fn[[Mat], Mat]:
    K: Mat = place_poles(A, B, poles).gain_matrix
    @jit(nopython=True)
    def c(x: Mat) -> Mat: return -K@x
    return c


# DELETE LATER
################################################################################
# Assignment parameters
A               = Mat([[5 - 0, 0.5 - 8], 
                       [0    , 1      ]])

B               = Mat([[0.0],
                       [1.0]])

A_inv           = Mat([[1.0/5.0, 1.5],
                      [0      , 1  ]])

K_bar           = Mat([[-16.0/3, 8.0]]) # solution to pole placement problem
###########################################################################

P_var = cp.Variable((2,2), symmetric=True)
Q_var = cp.Variable((2,2), symmetric=True)

problem = cp.Problem(cp.Minimize(0), [P_var - np.eye(2) >> 0,
                                      Q_var - np.eye(2) >> 0,
                                      (A-B@K_bar).T @ P_var + P_var @ (A-B@K_bar) == -Q_var])

problem.solve()
#print(P.value, Q.value)
P = P_var.value
Q = Q_var.value

def controller(xsk):
    return -K_bar@xsk


def create_triggering_function(sigma):
    def phi_e(x, eps):
        x = np.reshape(x, (2,1))
        eps = np.reshape(eps, (2,1))
        return np.block([[x.T, eps.T]])@np.block([[(1-sigma)*Q, P@B@K_bar],[(B@K_bar).T @ P, np.zeros((2,2))]])@np.block([[x],[eps]])

    def PHI(x, xsk):
        eps = xsk - x
        if phi_e(x, eps) <= 0:  return True 
        else:                   return False

    return PHI
