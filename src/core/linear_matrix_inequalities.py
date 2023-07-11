import cvxpy                as cp
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.ticker    as ticker

# from ..params.assignment_2_params import*
from numba      import jit
from copy       import copy
from typedefs   import *

# I hope no one ever reads this code, me included.

# TODO: DELETE LATER

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


P = cp.Variable((2,2), symmetric=True)
Q = cp.Variable((2,2), symmetric=True)
h = cp.Parameter((), value=0.0, pos=True)

def eA(h):
    return cp.bmat([[cp.exp(5.0*h), -cp.exp(h)*(15.0/8.0)*(cp.exp(4.0*h) - 1)],
                    [0.0          , cp.exp(h)]])

def G(h):
    return cp.bmat([[(15.0/8.0)*cp.exp(h)-(3.0/8.0)*cp.exp(5.0*h) - (3.0/2.0)],
                    [cp.exp(h) - 1]])

#################################################################
# Verifying results of derivations in Q1.2
#################################################################
def eA_np(h):
    return np.block([[np.exp(5.0*h), -np.exp(h)*(15.0/8.0)*(np.exp(4.0*h) - 1)],
                    [0.0          , np.exp(h)]])

def G_np(h):
    return np.block([[(15.0/8.0)*np.exp(h)-(3.0/8.0)*np.exp(5.0*h) - (3.0/2.0)],
                    [np.exp(h) - 1]])

def F_cl_0_to_hold_np(h):
    return np.block([[eA_np(h) - G_np(h) @ K_bar, Zeros2x1],
                     [-K_bar                    , Zeros1x1]])

def F_cl_1_to_hold_np(h):
    return np.block([[eA_np(h), G_np(h)],
                     [Zeros1x2, I1x1]]) 

def A_cl_to_hold_np(h): return F_cl_1_to_hold_np(h) @ F_cl_0_to_hold_np(h) @ F_cl_0_to_hold_np(h)  
def A_cl_to_zero_np(h): return eA_np(h) @ (eA_np(h) - G_np(h) @ K_bar) @ (eA_np(h) - G_np(h) @ K_bar) 

@jit(nopython=True)
def spectral_radius(A): return max([abs(x) for x in np.linalg.eigvals(A)])
#################################################################

# To hold
def F_cl_to_hold(h_n): return eA(h_n) - G(h_n) @ K_bar
def A_cl_to_hold_loss(h): return F_cl_to_hold(2*h) @ F_cl_to_hold(h) @ F_cl_to_hold(h)
def A_cl_to_hold_no_loss(h): return F_cl_to_hold(h) @ F_cl_to_hold(h) @ F_cl_to_hold(h)

# To zero
def F_cl_to_zero(h, h_n): return eA(h_n) - eA(h_n - h) @ G(h) @ K_bar
def A_cl_to_zero_loss(h): return F_cl_to_zero(h, 2*h) @ F_cl_to_zero(h, h) @ F_cl_to_zero(h, h)
def A_cl_to_zero_no_loss(h): return F_cl_to_zero(h, h) @ F_cl_to_zero(h, h) @ F_cl_to_zero(h, h)


# Question 2.3 LMIs
q2_to_hold_stability = cp.Problem(cp.Minimize(0), [P - I2x2 >> 0,
                                                   P - A_cl_to_hold_loss(h).T @ P @ A_cl_to_hold_loss(h) - I2x2 >> 0,
                                                   P - A_cl_to_hold_no_loss(h).T @ P @ A_cl_to_hold_no_loss(h) - I2x2 >> 0, 
                                                   ] )

q2_to_zero_stability = cp.Problem(cp.Minimize(0), [P - I2x2 >> 0,
                                                   P - A_cl_to_zero_loss(h).T @ P @ A_cl_to_zero_loss(h) - I2x2 >> 0,
                                                   P - A_cl_to_zero_no_loss(h).T @ P @ A_cl_to_zero_no_loss(h) - I2x2 >> 0,
                                                   ] )


# Question 3.3 LMIs
p0 = 0.01
p1 = 0.51
M0 = cp.Variable((2,2), symmetric=True)
M1 = cp.Variable((2,2), symmetric=True)

q3_to_hold_stability = cp.Problem(cp.Minimize(0), [M0 - I2x2 >> 0,
                                                   M1 - I2x2 >> 0,
                                                   M0 - (1-p0)*F_cl_to_hold(h).T @ M0 @ F_cl_to_hold(h) - p0*F_cl_to_hold(2*h).T @ M1 @ F_cl_to_hold(2*h) - I2x2 >> 0,
                                                   M1 - (1-p1)*F_cl_to_hold(h).T @ M0 @ F_cl_to_hold(h) - p1*F_cl_to_hold(2*h).T @ M1 @ F_cl_to_hold(2*h) - I2x2 >> 0,
                                                   ])

q3_to_zero_stability = cp.Problem(cp.Minimize(0), [M0 - I2x2 >> 0,
                                                   M1 - I2x2 >> 0,
                                                   M0 - (1-p0)*F_cl_to_zero(h, h).T @ M0 @ F_cl_to_zero(h,h) - p0*F_cl_to_zero(h, 2*h).T @ M1 @ F_cl_to_zero(h, 2*h) - I2x2 >> 0,
                                                   M1 - (1-p1)*F_cl_to_zero(h, h).T @ M0 @ F_cl_to_zero(h,h) - p1*F_cl_to_zero(h, 2*h).T @ M1 @ F_cl_to_zero(h, 2*h) - I2x2 >> 0,
                                                   ])

def H0(h):
    return cp.bmat([[cp.exp(5*h)-8 , (-15*cp.exp(h)*(cp.exp(4*h) - 1))/8.0 + (3/2) , (15*cp.exp(h)-3*cp.exp(5*h))/8.0],
                     [-(16/3.0)     , cp.exp(h)+8                                   , cp.exp(h)],
                     [(16/3.0)      , -8                                            , 0]])
def H1():
    return cp.bmat([[10        , -(15/8.0) , -(15/8.0)],
                     [(16/3.0)  , -8        , -1],
                     [0         , 0         , 0]])

def H2():
    return cp.bmat([[-2, (3/8.0)   , (3/8.0)],
                     [0 , 0         , 0],
                     [0 , 0         , 0]])

def Polytopic_closed_loop(h, alpha_1, alpha_2): return H0(h) + alpha_1*H1() + alpha_2*H2()

#P3x3        = cp.Variable((3,3), symmetric=True)
#alpha_1_min = cp.Parameter((), value=0.0)
#alpha_1_max = cp.Parameter((), value=0.0)
#alpha_2_min = cp.Parameter((), value=0.0)
#alpha_2_max = cp.Parameter((), value=0.0)

def q4_2_stability(h, alpha_1_min, alpha_2_min, alpha_1_max, alpha_2_max, P3x3):
    return cp.Problem(cp.Minimize(0), [P3x3 - np.eye(3) >> 0,
                                        P3x3 - Polytopic_closed_loop(h, alpha_1_max, alpha_2_max).T @ P3x3 @ Polytopic_closed_loop(h, alpha_1_max, alpha_2_max) - np.eye(3) >> 0,
                                        P3x3 - Polytopic_closed_loop(h, alpha_1_max, alpha_2_min).T @ P3x3 @ Polytopic_closed_loop(h, alpha_1_max, alpha_2_min) - np.eye(3) >> 0,
                                        P3x3 - Polytopic_closed_loop(h, alpha_1_min, alpha_2_max).T @ P3x3 @ Polytopic_closed_loop(h, alpha_1_min, alpha_2_max) - np.eye(3) >> 0,
                                        P3x3 - Polytopic_closed_loop(h, alpha_1_min, alpha_2_min).T @ P3x3 @ Polytopic_closed_loop(h, alpha_1_min, alpha_2_min) - np.eye(3) >> 0,
                                        ])

def q4_2_stability_refined(alpha_1_min_1, alpha_2_min_1, alpha_1_max_1, alpha_2_max_1,
                           alpha_1_min_2, alpha_2_min_2, alpha_1_max_2, alpha_2_max_2,
                           alpha_1_min_3, alpha_2_min_3, alpha_1_max_3, alpha_2_max_3,
                           h, P3x3_1, P3x3_2, P3x3_3): # P3x3_2 and P3x3_3 are not used, since they shouldn't be
    return cp.Problem(cp.Minimize(0), [ P3x3_1 - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_1, alpha_2_max_1).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_1, alpha_2_max_1) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_1, alpha_2_min_1).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_1, alpha_2_min_1) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_1, alpha_2_max_1).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_1, alpha_2_max_1) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_1, alpha_2_min_1).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_1, alpha_2_min_1) - np.eye(3) >> 0,
                                        
                                        #P3x3_2 - np.eye(3) >> 0, # think I should only be using one P
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_2, alpha_2_max_2).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_2, alpha_2_max_2) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_2, alpha_2_min_2).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_2, alpha_2_min_2) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_2, alpha_2_max_2).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_2, alpha_2_max_2) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_2, alpha_2_min_2).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_2, alpha_2_min_2) - np.eye(3) >> 0,
                                        
                                        #P3x3_3 - np.eye(3) >> 0, # think I should only be using one P
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_3, alpha_2_max_3).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_3, alpha_2_max_3) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_max_3, alpha_2_min_3).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_max_3, alpha_2_min_3) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_3, alpha_2_max_3).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_3, alpha_2_max_3) - np.eye(3) >> 0,
                                        P3x3_1 - Polytopic_closed_loop(h, alpha_1_min_3, alpha_2_min_3).T @ P3x3_1 @ Polytopic_closed_loop(h, alpha_1_min_3, alpha_2_min_3) - np.eye(3) >> 0,


                                       ])

# fuck it, I'm just doing this...
TASK = int(input("\n\n\nwhich one? (1, 2, 3, 4)\n\n\n"))

if TASK == 1: # just for numerical verification of symbolic results in MATLAB etc.
    spectral_radius_vales = []
    h_values = np.linspace(0+0.000001, 5.0, 1000)
    for h_value in h_values:
        h.value = copy(h_value) # copy() cause' got paranoid when I was having trouble getting the correct matrix expressions
        spectral_radius_vales.append(spectral_radius(A_cl_to_hold_np(h_value))) 
    plt.plot(h_values, spectral_radius_vales)
    plt.plot(h_values, [1]*len(h_values))
    plt.yscale("log")
    plt.show()

if TASK == 2: 
    problem_to_hold = q2_to_hold_stability
    problem_to_zero = q2_to_zero_stability
    h_values = np.linspace(0+0.000001, 10.0, 1000)
    to_hold_edges = [0.0, 0.2389392, 0.2526, 0.2761] # hardcoded garbage
    to_zero_edges = [0.0, 0.19059, 0.21441, 0.28378] # hardcoded garbage
    
if TASK == 3: 
    problem_to_hold = q3_to_hold_stability
    problem_to_zero = q3_to_zero_stability
    h_values = np.linspace(0.0+0.000001, 20.0, 5000)
    to_hold_edges = [0.0, 0.1658] # hardcoded garbage
    to_zero_edges = [0.0, 0.2354] # hardcoded garbage

h_stable_to_hold = []
h_stable_to_zero = []

h_failed_to_hold = []
h_failed_to_zero = []

if TASK in [2,3]: # TODO: parallelize this as well
    for h_value in h_values:
        h.value = copy(h_value)

        try:
            problem_to_hold.solve()
            if problem_to_hold.status in ["optimal", "optimal_inaccurate"] : h_stable_to_hold.append(h_value)
        except:
            h_failed_to_hold.append(h_value)

        try:
            problem_to_zero.solve()
            if problem_to_zero.status in ["optimal", "optimal_inaccurate"]: h_stable_to_zero.append(h_value)
        except ValueError:
            h_failed_to_zero.append(h_value)

# This is all very fucking ugly, but I have spent more than too much time on this
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.patch.set_alpha(0.0)

    vline_height = 0.1
    plt.vlines(to_hold_edges, 1-vline_height, 1+vline_height, color="orange", linewidth=2)
    plt.scatter(h_stable_to_hold, [1]*len(h_stable_to_hold), marker="_", linewidth=2, color="orange", label="'To-hold'")

    plt.vlines(to_zero_edges, 0-vline_height, 0+vline_height, color="blue", linewidth=2)
    plt.scatter(h_stable_to_zero, [0]*len(h_stable_to_zero), marker="_", linewidth=2, color = "blue", label="'To-zero'")

    if len(h_failed_to_hold) + len(h_failed_to_zero) != 0:
        plt.scatter(h_failed_to_hold+h_failed_to_zero, [1]*len(h_failed_to_hold)+[0]*len(h_failed_to_zero), marker="X", linewidth=2, color="red", label="Failed")

    ax.set_ylim(-0.6, 1.6)
    ax2 = ax.secondary_xaxis("top")
    plt.subplots_adjust(bottom=0.2)
    ax3 = ax.secondary_xaxis(-0.06)
    ax.xaxis.set_major_locator(ticker.FixedLocator(to_zero_edges))
    ax2.xaxis.set_major_locator(ticker.FixedLocator(to_hold_edges))
    ax3.xaxis.set_major_locator(ticker.AutoLocator())
    ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.set_xlabel("h", fontsize=18)

    ax.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax3.tick_params(axis="x", labelsize=12)

    plt.legend(fontsize=14)
    plt.show()



# -------------------------------------------------------
# Crimes against humanity below. Humanitarian aid needed.
# -------------------------------------------------------


if TASK == 4:
    REFINE = False
    u_input = input("\n\n\nRefine polytope? (y OR yes OR 1 OR true):")
    if str(u_input) in ["y", "Y", "yes", "Yes", "1", "true", "True"]: REFINE = True
    print("\nREFINE =", REFINE)

    h_steps = 40 
    h_min   = 0.0 + 0.000001
    h_max   = 0.15 
    tau_min = 0.0
    h_values = np.linspace(h_min, h_max, h_steps)

    alphas_to_check = []
    points_to_check = []
    for h_value in h_values:
        h.value             = h_value
    
        tau_max             = h_value
        tau_steps           = (float(tau_max - tau_min) / float(h_max - h_min)) * float(h_steps)
        one_step            = float(tau_max - tau_min) / float(tau_steps) if tau_steps != 0.0 else 0.0
        correction          = float(tau_steps - int(tau_steps)) * one_step # i want a nice grid ok?

        #print("h_value, tau_max, tau_steps:", h_value, tau_max, tau_steps)
        for tau_min_this_iter in np.linspace(tau_min, tau_max - one_step - correction, int(tau_steps)):
            
            tau_steps_this_iter     = (float(tau_max - tau_min_this_iter) / float(h_max - h_min)) * float(h_steps)
            one_step_this_iter      = float(tau_max - tau_min_this_iter) / float(tau_steps_this_iter) if tau_steps_this_iter != 0.0 else 0.0
            correction_this_iter    = float(tau_steps_this_iter - int(tau_steps_this_iter)) * one_step_this_iter # i want a nice grid ok?

            _alpha_1_max = np.exp(h_value - tau_min_this_iter) 
            _alpha_2_max = np.exp(5*(h_value - tau_min_this_iter)) 

            #print("tau_min_this_iter, tau_steps_this_iter:", tau_min_this_iter, tau_steps_this_iter) 
            for tau_max_this_iter in np.linspace(tau_min_this_iter, tau_max - one_step_this_iter - correction_this_iter, int(tau_steps_this_iter)): # exclusive end range

                _alpha_1_min = np.exp(h_value - tau_max_this_iter)
                _alpha_2_min = np.exp(5*(h_value - tau_max_this_iter))

                alphas_to_check.append((_alpha_1_min, _alpha_2_min, _alpha_1_max, _alpha_2_max)) # precompute points to check for convenience
                points_to_check.append((h_value, tau_min_this_iter, tau_max_this_iter))          # precompute points to check for convenience

    n = len(points_to_check)
    print("\n\nNum. points to check =", n, "\n\n")
    if n > 10000: print("\n\nthis will take some time...\n\n")

    # TODO: move all this garbage elsewhere
    from multiprocessing import Pool
    from os import cpu_count
    logical_processors = cpu_count()
    print("cpu_count(): ", logical_processors)

    
    def run_this_fucking_garbage(points_and_alphas_tuple):
        stable_points = []
        points_to_check, alphas_to_check = points_and_alphas_tuple
        P3x3        = cp.Variable((3,3), symmetric=True)
        alpha_1_min = cp.Parameter((), value=0.0)
        alpha_1_max = cp.Parameter((), value=0.0)
        alpha_2_min = cp.Parameter((), value=0.0)
        alpha_2_max = cp.Parameter((), value=0.0)
        h           = cp.Parameter((), value=0.0)

        q4_2_problem = q4_2_stability(h, alpha_1_min, alpha_2_min, alpha_1_max, alpha_2_max, P3x3) 

        def set_alphas(_alpha_1_min, _alpha_2_min, _alpha_1_max, _alpha_2_max):
            alpha_1_min.value = _alpha_1_min
            alpha_2_min.value = _alpha_2_min
            alpha_1_max.value = _alpha_1_max
            alpha_2_max.value = _alpha_2_max

        local_n = len(points_to_check)
        for i, (point, alphas) in enumerate(zip(points_to_check, alphas_to_check)):
            if i % 200 == 0: print("\nworker at iteration:",i,"of",local_n,"\n")
            h.value = point[0]
            set_alphas(*alphas)
            q4_2_problem.solve()
            if q4_2_problem.status in ["optimal", "optimal_inaccurate"]:
                stable_points.append(point)

        return stable_points
    
    # refined garbage
    def run_this_fucking_garbage_refined(points_and_alphas_tuple):
        stable_points = []
        points_to_check, alphas_to_check = points_and_alphas_tuple

        P3x3_1        = cp.Variable((3,3), symmetric=True)
        alpha_1_min_1 = cp.Parameter((), value=0.0)
        alpha_1_max_1 = cp.Parameter((), value=0.0)
        alpha_2_min_1 = cp.Parameter((), value=0.0)
        alpha_2_max_1 = cp.Parameter((), value=0.0)

        P3x3_2        = cp.Variable((3,3), symmetric=True)
        alpha_1_min_2 = cp.Parameter((), value=0.0)
        alpha_1_max_2 = cp.Parameter((), value=0.0)
        alpha_2_min_2 = cp.Parameter((), value=0.0)
        alpha_2_max_2 = cp.Parameter((), value=0.0)

        P3x3_3        = cp.Variable((3,3), symmetric=True)
        alpha_1_min_3 = cp.Parameter((), value=0.0)
        alpha_1_max_3 = cp.Parameter((), value=0.0)
        alpha_2_min_3 = cp.Parameter((), value=0.0)
        alpha_2_max_3 = cp.Parameter((), value=0.0)

        h           = cp.Parameter((), value=0.0)

        q4_2_problem = q4_2_stability_refined(alpha_1_min_1, alpha_2_min_1, alpha_1_max_1, alpha_2_max_1,
                                              alpha_1_min_2, alpha_2_min_2, alpha_1_max_2, alpha_2_max_2,
                                              alpha_1_min_3, alpha_2_min_3, alpha_1_max_3, alpha_2_max_3,
                                              h, P3x3_1, P3x3_2, P3x3_3) 

        def set_alphas(point):
            h, tau_min, tau_max = point

            tau_min_1 = tau_min
            tau_max_1 = tau_min + (tau_max - tau_min)/3.0

            tau_min_2 = 1.0*tau_max_1
            tau_max_2 = tau_min + 2*(tau_max - tau_min)/3.0

            tau_min_3 = 1.0*tau_max_2
            tau_max_3 = tau_max

            # THESE ARE NOT ALL NECESSARY, 4 OF THEM ARE REDUNDANT

            alpha_1_min_1.value = np.exp(h - tau_max_1)
            alpha_2_min_1.value = np.exp(5*(h - tau_max_1))
            alpha_1_max_1.value = np.exp(h - tau_min_1)
            alpha_2_max_1.value = np.exp(5*(h - tau_min_1))
            
            alpha_1_min_2.value = np.exp(h - tau_max_2)
            alpha_2_min_2.value = np.exp(5*(h - tau_max_2))
            alpha_1_max_2.value = np.exp(h - tau_min_2)
            alpha_2_max_2.value = np.exp(5*(h - tau_min_2))
            
            alpha_1_min_3.value = np.exp(h - tau_max_3)
            alpha_2_min_3.value = np.exp(5*(h - tau_max_3))
            alpha_1_max_3.value = np.exp(h - tau_min_3)
            alpha_2_max_3.value = np.exp(5*(h - tau_min_3))

        local_n = len(points_to_check)
        for i, (point, alphas) in enumerate(zip(points_to_check, alphas_to_check)): # alphas are unused, see if I care
            if i % 200 == 0: print("\nworker at iteration:",i,"of",local_n,"\n")

            h.value = point[0]
            set_alphas(point)
            q4_2_problem.solve()
            if q4_2_problem.status in ["optimal", "optimal_inaccurate"]:
                stable_points.append(point)

        return stable_points

    def flatten(l):
        return [item for sublist in l for item in sublist]

    
    print("\nstarting workers...\n")
    with Pool() as pool:
        inputs = [(array1.tolist(), array2.tolist()) for array1, array2 in zip(np.array_split(points_to_check, logical_processors), np.array_split(alphas_to_check, logical_processors))]
        stable_points = flatten(list(pool.map(run_this_fucking_garbage, inputs)))
        if REFINE: 
            print("\n\n\n\n\nnow doing the same for refined polytope\n\n\n\n\n")
            stable_points_2 = flatten(list(pool.map(run_this_fucking_garbage_refined, inputs)))
    print("\nwork done.\n")

    print("\n\nstable_points:\n\n", stable_points,"\n\n")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    h_list          = list(list(zip(*stable_points))[0])
    tau_min_list    = list(list(zip(*stable_points))[1])
    tau_max_list    = list(list(zip(*stable_points))[2])
    ax.scatter(h_list, tau_min_list, tau_max_list, label="stable points", color="blue", s=25, alpha=0.7)

    if REFINE:
        h_list_r          = list(list(zip(*stable_points_2))[0])
        tau_min_list_r    = list(list(zip(*stable_points_2))[1])
        tau_max_list_r    = list(list(zip(*stable_points_2))[2])
        ax.scatter(h_list_r, tau_min_list_r, tau_max_list_r, label="stable points (refined)", color="green", s=25, alpha=0.7)

    h_list_2          = list(list(zip(*points_to_check))[0])
    tau_min_list_2    = list(list(zip(*points_to_check))[1])
    tau_max_list_2    = list(list(zip(*points_to_check))[2])
    ax.scatter(h_list_2, tau_min_list_2, tau_max_list_2, label="checked points", color="red", s=2, alpha=0.3)


    ax.set_xlabel('h')
    ax.set_ylabel('tau_min')
    ax.set_zlabel('tau_max')
    plt.legend()
    plt.show()
