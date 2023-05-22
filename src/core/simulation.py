# TODO
from .typedefs import *
from .dynamic_system import SampledDataNCS
from numba import jit
from numpy import zeros, float64
from matplotlib.pyplot import plot, show, gcf 

"""
Simulate a SampledDataNCS system from x0, t0 to tf using simulation stepsize dt.
Simulation method is ERK4. The trajectory is returned as a matrix with the first column being time,
and subsequent columns being state trajectories.
"""
def simulate(sys: SampledDataNCS, x0: Mat, t0: float, tf: float, dt: float):
    N: int = int((tf-t0)/dt)
    trajectory: Mat = zeros((N, x0.shape[0]+1), dtype = float64)
    sys.t = t0
    sys.x = x0
    for i in range(N):
        trajectory[i][::] = [sys.t, *sys.x]
        sys.u = sys.a(sys.c(sys.s(sys.x)))
        sys.t, sys.x = erk4_step(sys.t, sys.f, sys.x, sys.u, dt)
    return trajectory

"""Basic Explicit 4th-Order Runge-Kutta Step."""
@jit(nopython=True)
def erk4_step(t: float, f: fn, x: Mat, u: Mat, dt: float) -> (float, Mat):
    k1: Mat = f(t, x, u)
    k2: Mat = f(t + 0.5*dt, x + 0.5*dt*k1, u)
    k3: Mat = f(t + 0.5*dt, x + 0.5*dt*k2, u)
    k4: Mat = f(t + dt, x + dt*k3, u)
    return t + dt, x + (1/6)*(k1 + 2.0*k2 + 2.0*k3 + k4)

"""Simple function for plotting the simulation trajectory."""
def plot_trajectory(trajectory: Mat):
    t: Mat = trajectory[:, 0]
    x: Mat = trajectory[:, 1:]
    plot(t, x, label=["x_1","x_2"])
    gcf().suptitle("Simulated Trajectory")
    gcf().legend()
    show()
