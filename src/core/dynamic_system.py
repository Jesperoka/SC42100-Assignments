# TODO: 
from .typedefs import * 
from scipy.signal import cont2discrete
from numba import jit

@jit(nopython=True)
def passthrough(anything):
    return anything

class SampledDataNCS:
    f: fn   # plant    
    c: fn   # controller
    s: fn   # sensor
    a: fn   # actuator

    t: float    # current time
    x: Mat      # current state
    u: Mat = 0  # current control input

    xsk: float      # state at last sample
    trig: fn        # triggering function
    comms: int = 0  # number of times trig has been triggered

    tau_sc: float # time delay from sensor to controller
    tau_ca: float # time delay within controller
    tau_c:  float # time delay from controller to actuator

    h_type: str # type of sampling interval
    h: float    # sampling interval length

    def __init__(self, f, c, a = passthrough, s = passthrough, trig = True, tau_sc = 0.0, tau_ca = 0.0, tau_c = 0.0, h_type = "constant"):
        self.f      = f
        self.c      = c
        self.a      = a
        self.s      = s
        self.trig   = trig
        self.tau_sc = tau_sc
        self.tau_ca = tau_ca
        self.tau_c  = tau_c
        self.h_type = h_type


def create_lti_system(A: Mat, B: Mat) -> fn:
    @jit(nopython=True)
    def f(t: float, x: Mat, u: Mat): return A@x + B@u
    return f

def create_discrete_lti_system(A: Mat, B: Mat, h) -> fn:
    Ad, Bd, _, _, _  = cont2discrete((A, B, Zeros1x2, Zeros2x1), h)
    @jit(nopython=True)
    def fd(x, u): return Ad@x + Bd@u
    return fd
