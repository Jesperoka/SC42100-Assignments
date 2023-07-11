# TODO
from .typedefs import *
from .dynamic_system import SampledDataNCS
from numba import jit
from numpy import zeros, float64, squeeze, reshape
from matplotlib.pyplot import plot, show, gcf, grid
from scipy.optimize import fsolve

"""
Simulate a SampledDataNCS system from x0, t0 to tf using simulation stepsize dt.
Simulation method is ERK4. The trajectory is returned as a matrix with the first column being time,
and subsequent columns being state trajectories.
"""
def simulate(sys: SampledDataNCS, x0: Mat, t0: float, tf: float, dt: float, h_PETC = None):
    N: int = int((tf-t0)/dt)
    trajectory: Mat = zeros((N, x0.shape[0]+1), dtype = float64)
    x0 = reshape(x0, (2,1))
    sys.t = t0
    sys.x = x0
    sys.xsk = x0

    def PETC(i): return i % int(h_PETC/dt) == 0
    if h_PETC is not None: petc = PETC
    else: petc = lambda i: True

    for i in range(N):
        trajectory[i][::] = [sys.t, *sys.x]
        if petc(i) and sys.trig(sys.s(sys.x), sys.xsk):
            sys.xsk = sys.s(sys.x)
            sys.comms += 1
        sys.u = sys.a(sys.c(sys.xsk))
        sys.t, sys.x = irk1_step(sys.t, sys.f, sys.x, sys.u, dt) #erk4_step(sys.t, sys.f, sys.x, sys.u, dt)
    return trajectory


"""Basic Explicit 4th-Order Runge-Kutta Step."""
@jit(nopython=True)
def erk4_step(t: float, f: fn, x: Mat, u: Mat, dt: float) -> (float, Mat):
    k1: Mat = f(t, x, u)
    k2: Mat = f(t + 0.5*dt, x + 0.5*dt*k1, u)
    k3: Mat = f(t + 0.5*dt, x + 0.5*dt*k2, u)
    k4: Mat = f(t + dt, x + dt*k3, u)
    return t + dt, x + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

"""Midpoint Rule"""
def irk1_step(t, f, x, u, dt):
    def midpoint(x_next):
        x_next = reshape(x_next, (2,1))
        return (x_next - (x + dt*f(t + 0.5*dt, 0.5*(x + x_next), u))).ravel()
    return t + dt, reshape(fsolve(midpoint, x.ravel()), (2,1))

"""Simple function for plotting the simulation trajectory."""
def plot_trajectory(trajectory: Mat):
    t: Mat = trajectory[:, 0]
    x: Mat = trajectory[:, 1:]
    plot(t, x, label=["x_1","x_2"])
    gcf().suptitle("Simulated Trajectory")
    gcf().legend()
    show()

def plot_trajectories(trajs_colors_and_sigmas: list[Mat]):
    for traj, color, sigma in trajs_colors_and_sigmas:

        t: Mat = traj[:, 0]
        x1: Mat = traj[:, 1]
        plot(t, x1, label="sigma = "+str(sigma), color=color)

    #gcf().suptitle("Simulated Trajectories")
    gcf().legend()
    grid(True)
    show()
