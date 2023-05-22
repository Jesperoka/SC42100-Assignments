"""Numerical analysis of systems"""
import pdb
from numpy import reshape, array, empty, nan, linspace, full, unique, exp, diag, csingle
from scipy.linalg import expm
from numpy.linalg import inv, eigvals, eig
from scipy.optimize import differential_evolution
from numba import jit, objmode, typeof
from copy import copy
from functools import partial
from matplotlib.pyplot import scatter, show, gcf, gca 

from .typedefs import *


def create_NCS_small_delay_closed_loop_matrix(tau: float, h: float, A: Mat, B: Mat, K: Mat, A_inv: Mat, P=None, D=None, P_inv=None) -> Mat:

    if D is not None: # then we assume A is diagonalizable and P, D, P_inv has been given
        eA_h: Mat           = P@(diag(diag(exp(D*h))).astype(number_type))@P_inv
        eA_tau: Mat         = P@(diag(diag(exp(D*tau))).astype(number_type))@P_inv
        eA_h_minus_tau: Mat = P@(diag(diag(exp(D*(h - tau)))).astype(number_type))@P_inv

    else: # otherwise we can't jit the scipy matrix exponential function
        with objmode(eA_h="Array(complex64, 2, 'C')", eA_tau="Array(complex64, 2, 'C')", eA_h_minus_tau="Array(complex64, 2, 'C')"):

            eA_h: Mat           = expm(A*h)
            eA_tau: Mat         = expm(A*tau)
            eA_h_minus_tau: Mat = expm(A*(h-tau))

    F: Mat      = BlkMat2D([(eA_h    , eA_h_minus_tau@A_inv@(eA_tau - I2x2)@B),
                            (Zeros1x2, Zeros1x1)])

    G: Mat      = BlkMat2D([(A_inv@(eA_h_minus_tau - I2x2)@B,),
                            (I1x1,)])

    return F - G@K

# assumes 0 <= tau < 2h
def create_NCS_large_delay_closed_loop_matrix(tau: float, h: float, A: Mat, B: Mat, K: Mat, A_inv: Mat, P=None, D=None, P_inv=None) -> Mat:
    
    if D is not None: # then we assume A is diagonalizable and P, D, P_inv has been given
        
        eA_h: Mat               = P@(diag(diag(exp(D*h))).astype(number_type))@P_inv
        eA_tau: Mat             = P@(diag(diag(exp(D*tau))).astype(number_type))@P_inv
        eA_h_minus_tau: Mat     = P@(diag(diag(exp(D*(h - tau)))).astype(number_type))@P_inv
        eA_tau_minus_2h: Mat    = P@(diag(diag(exp(D*(tau - 2*h)))).astype(number_type))@P_inv
        eA_2h_minus_tau: Mat    = P@(diag(diag(exp(D*(2*h - tau)))).astype(number_type))@P_inv

    else: # otherwise we can't jit the scipy matrix exponential function
        with objmode(eA_h="Array(complex64, 2, 'C')", eA_tau="Array(complex64, 2, 'C')", eA_h_minus_tau="Array(complex64, 2, 'C')", eA_tau_minus_2h="Array(complex64, 2, 'C')", eA_2h_minus_tau="Array(complex64, 2, 'C')"):

            eA_h: Mat               = expm(A*h)
            eA_tau: Mat             = expm(A*tau)
            eA_h_minus_tau: Mat     = expm(A*(h - tau))
            eA_tau_minus_2h: Mat    = expm(A*(tau - 2*h))
            eA_2h_minus_tau: Mat    = expm(A*(2*h - tau))
    
#    breakpoint()
    f12 = eA_h_minus_tau@A_inv@(eA_tau - I2x2)@B if tau < h else -eA_2h_minus_tau@A_inv@(eA_tau_minus_2h - I2x2)@B 
    f13 = Zeros2x1                               if tau < h else -eA_h@A_inv@(eA_h_minus_tau - I2x2)@B
    g1  = A_inv@(eA_h_minus_tau - I2x2)@B        if tau < h else Zeros2x1

    F: Mat = BlkMat2D([(eA_h    , f12     , f13),
                       (Zeros1x2, Zeros1x1, Zeros1x1),
                       (Zeros1x2, I1x1    , Zeros1x1)])

    G: Mat = BlkMat2D([(g1,),
                       (I1x1,),
                       (Zeros1x1,)])

    return F - G@K

#@jit(nopython=True)
def is_asymptotically_stable_discrete(A: Mat) -> bool:
    return spectral_radius(A) < 1.0

@jit(nopython=True)
def spectral_radius(A: Mat):
    eigenvalues: Mat = eigvals(A)
    return inf_norm(eigenvalues) 

@jit(nopython=True) # TODO: use numpy inf norm instead
def inf_norm(vector) -> float:
    return max([abs(value) for value in vector])


def investigate_stability(A: Mat, B: Mat, K: Mat,
                          h_range:           (float, float) = (0.0, 1.0),
                          h_steps:                      int = 300,
                          tau_range_function:           fn = lambda h: (0.0, h),
                          matrix_creator_function:      fn = lambda *args: Zeros2x2,
                          stability_checker_function:   fn = lambda x: False):

    assert h_range[1] > h_range[0] and h_range[0] >= 0.0 and h_steps >= 1 
    A_inv = inv(A)

    # Diagonalize A, if we can, for significantly faster matrix exponential calculations
    eigenvalues, eigenvectors = eig(A)
    if unique(eigenvalues).size == eigenvalues.size: # this check is sufficient but not necessary
        P       = eigenvectors.astype(number_type) 
        D       = diag(eigenvalues).astype(number_type)
        P_inv   = inv(P).astype(number_type)
    else:
        P       = None
        D       = None
        P_inv   = None


    matrix_creator_function     = jit(matrix_creator_function, nopython=True)
    stability_checker_function  = jit(stability_checker_function, nopython=True)
    tau_range_function          = jit(tau_range_function, nopython=True)
    
    @jit(nopython=True)
    def run():

        h_points            = full((2*h_steps*h_steps, 1), nan) # 2*... is hardcoded size limit for tau range
        tau_points          = full((2*h_steps*h_steps, 1), nan) # 2*... is hardcoded size limit for tau range
   
        i: int = 0
        for h in linspace(h_range[0], h_range[1], int(h_steps)): # inclusive end range

            tau_min, tau_max    = tau_range_function(h)
            tau_steps           = (float(tau_max - tau_min) / float(h_range[1] - h_range[0])) * float(h_steps)
            one_step            = float(tau_max - tau_min) / float(tau_steps) if tau_steps != 0.0 else 0.0
            correction          = float(tau_steps - int(tau_steps)) * one_step # I want a nice grid ok?

            for tau in linspace(tau_min, tau_max - one_step - correction, int(tau_steps)): # exclusive end range

                closed_loop_matrix = matrix_creator_function(tau, h, A, B, K, A_inv, P=P, D=D, P_inv=P_inv)
                if stability_checker_function(closed_loop_matrix):

                    h_points[i]     = h
                    tau_points[i]   = tau

                    i += 1

        return (h_points, tau_points)

    return run()


def max_acceptable_tau(h: float, h_steps: int, A: Mat, B: Mat, K_shape: tuple, K: Mat):
    K = reshape(K, K_shape) 
    tau_max: float = 0.0
    tau: float = 0.0
    tau_stepsize: float = h / h_steps

    A = A.astype(number_type)
    B = B.astype(number_type)
    K = K.astype(number_type)

    A_inv = inv(A)

    # Diagonalize A, if we can, for significantly faster matrix exponential calculations
    eigenvalues, eigenvectors = eig(A)
    if unique(eigenvalues).size == eigenvalues.size: # this check is sufficient but not necessary
        P       = eigenvectors.astype(number_type) 
        D       = diag(eigenvalues).astype(number_type)
        P_inv   = inv(P).astype(number_type)
    else:
        P       = None
        D       = None
        P_inv   = None

    while tau < h:
        closed_loop_matrix = create_NCS_small_delay_closed_loop_matrix(tau, h, A, B, K, A_inv, P=P, D=D, P_inv=P_inv)
        if is_asymptotically_stable_discrete(closed_loop_matrix):
            tau_max = copy(tau)
            tau += tau_stepsize
        else: 
            break

    return -tau_max # minimizing

seed = 69420 
def optimize_discrete_feedback(h: float, h_steps: int, A: Mat, B: Mat, K0: Mat):
    f = partial(max_acceptable_tau, h, h_steps, A, B, K0.shape)
    k_min = -50
    k_max =  50
    bounds = [(k_min, k_max) for _ in range(max(K0.shape))]
    result = differential_evolution(f, bounds, 
                                    maxiter=1000, 
                                    popsize=15, 
                                    x0=K0, 
                                    tol=0.00005, 
                                    mutation=(0.5, 1),
                                    recombination=0.7,
                                    seed=seed)
    return result






# ----------------------- Plotting -----------------------


def display_h_tau_results(h_points, tau_points, title: str):
    scatter(h_points, tau_points, label="Stable Points ($h$, $\\tau$)")
    figure = gcf()
    figure.gca().set_xlabel("$h$ [time]")
    figure.gca().set_ylabel("$\\tau$ [time]")
    figure.suptitle(title)
    figure.legend()
    show()

