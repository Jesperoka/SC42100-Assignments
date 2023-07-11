from .params.assignment_2_params import *
from .core.simulation import *
from .core.dynamic_system import *
from .core.controllers import *
from numpy.random import rand, seed

def main():
    seed(96024)

    x0s = 10.0*rand(5, 2) - 5.0

    t0 = 0
    tf = 5
    dt = 0.001

    sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ["blue", "orange", "green", "red", "purple"]

    h_PETC = 0.5*0.281

    f = create_lti_system(A, B)
    c = controller
    
    trajs_colors_and_sigmas = []
    avg_comms = []

    FINAL_TASK = True if str(input("final task?[y/n]")) in ["y", "Y", "yes", "Yes", "YES"] else False
    if not FINAL_TASK:
        for (sigma, color) in zip(sigmas, colors):
            avg_comm = 0
            for x0 in x0s:
                trig = create_triggering_function(sigma)
                system = SampledDataNCS(f=f, c=c, trig=trig)
                trajs_colors_and_sigmas.append((simulate(system, x0, t0, tf, dt, h_PETC=h_PETC), color, sigma))
                avg_comm += system.comms / len(x0s)

            avg_comms.append(avg_comm)

        print(avg_comms)

        plot_trajectories(trajs_colors_and_sigmas)

    if FINAL_TASK: # its 06:34 am, and deadline is 09:00
        for x0 in x0s:
            trig1 = create_triggering_function(0.9)
            system1 = SampledDataNCS(f=f, c=c, trig=trig1)
            trajs_colors_and_sigmas.append((simulate(system1, x0, t0, tf, dt, h_PETC=None), "blue", 0.9))

        for x0 in x0s:
            trig2 = create_triggering_function(0.9)
            system2 = SampledDataNCS(f=f, c=c, trig=trig2)
            trajs_colors_and_sigmas.append((simulate(system2, x0, t0, tf, dt, h_PETC=h_PETC), "orange", 0.9))

        plot_trajectories(trajs_colors_and_sigmas)

    return 0
