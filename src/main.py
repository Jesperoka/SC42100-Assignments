from .params.assignment_1_params import *
from .core.analysis import * 
from .core.simulation import * 


def main():

    user_input = welcome_user()

    # TODO: move params elsewhere
    h_range = (0.0, 5.0)
    h_steps = 1000

    match user_input: # TODO: fix arguments to functions
        case 1:
            print("\nInvestigating stability of NCS small delay closed loop system...")
            points = investigate_stability(A, B, K_static_Q2, 
                                           h_range=h_range, 
                                           h_steps=h_steps, 
                                           tau_range_function=lambda _h: (0.0, _h), 
                                           matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                           stability_checker_function=is_asymptotically_stable_discrete)  
            display_h_tau_results(points[0], points[1], "Stability of ($h$, $\\tau$) Combinations for Fixed $\\tau < h$") 

        case 2:
            h = 0.25
            print("\nOptimizing discrete model full state feedback for [0, tau_max] to be as large as possible for h = "+str(h)+".")
            result = optimize_discrete_feedback(h, h_steps, A, B, K_static_Q2)
            print("\nResult of optimization:\n", result, "\nInvestigating stability of resulting NCS small delay closed loop system...")
            points = investigate_stability(A, B, Mat([result.x]), 
                                           h_range=h_range, 
                                           h_steps=h_steps, 
                                           tau_range_function=lambda _h: (0.0, _h), 
                                           matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                           stability_checker_function=is_asymptotically_stable_discrete)  
            display_h_tau_results(points[0], points[1], "Stability of ($h$, $\\tau$) Combinations for Fixed $\\tau < h$") 

        case 3:
            # h = 0.25
            print("\nInvestigating stability of NCS large delay closed loop system...")
            points = investigate_stability(A, B, K_static_Q3, 
                                           h_range=h_range, 
                                           h_steps=h_steps, 
                                           tau_range_function=lambda _h: (0.0*_h, 1.5*_h), 
                                           matrix_creator_function=create_NCS_large_delay_closed_loop_matrix, 
                                           stability_checker_function=is_asymptotically_stable_discrete)  
            display_h_tau_results(points[0], points[1], "Stability of ($h$, $\\tau$) Combinations for Fixed $0 < \\tau < 1.5h$") 

        case 4:
            pass
        case 5:
            pass
        case _:
            pass

    return 0






def welcome_user() -> int:
    print("\n -- What to run? -- \n")
    print("""
          \n1: Question 2.2 (h, tau)-plot
          \n2: Question 2.3 K-optimization and resulting (h, tau)-plot
          \n3: Question 3.2 (h, tau)-plot
          \n4: Question 3.3 K-optimization and resulting (h, tau)-plot
          \n5: 
          """)

    user_input = int(input(""))
    in_list = [1,2,3,4,5]
    while user_input not in in_list:
        user_input = int(input("Invalid input.\n\nValid inputs are: " + str(in_list)))

    return user_input
