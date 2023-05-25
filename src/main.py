from .params.assignment_1_params import *
from .core.analysis import * 


def main():

    user_input = welcome_user()

    # TODO: move params elsewhere
    h_range = (0.0, 5.0)
    h_steps = 1000

    match user_input: 

        case 1:
            print("\n\n\nInvestigating stability of NCS small delay closed loop system with K_bar_1...\n\n\n")

            points1 = investigate_stability(A, B, K_static_Q2, 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            tau_range_function=lambda _h: (0.0, _h), 
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete)

            print("\n\n\nInvestigating stability of NCS small delay closed loop system with K_bar_2...\n\n\n")

            points2 = investigate_stability(A, B, K_static_Q2_2, 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            tau_range_function=lambda _h: (0.0, _h), 
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete)

            display_multiple_h_tau_results([(points1[0], points1[1]), (points2[0], points2[1])],
                                           ["Stable $(h, \\tau)$-points Static Contr. $\\bar{K}_1$", "Stable $(h, \\tau)$-points Static Contr. $\\bar{K}_2$"],
                                           "Stability of ($h$, $\\tau$) Combinations for Fixed $0 \\leq \\tau < h$") 

        case 2:

            print("\n\n\nOptimizing discrete model full state feedback for [0, tau_max] to be as large as possible for h = "+str(h_Q2_2)+".\n\n\n")

            result = optimize_discrete_feedback(K_static_Q2, -50, 50, max_stable_tau,
                                                h_Q2_2, A, B, K_static_Q2.shape, 
                                                matrix_creator_function=create_NCS_small_delay_closed_loop_matrix,
                                                stability_checker_function=is_asymptotically_stable_discrete,
                                                tau_range_function=lambda _h: (0.0, _h))

            print("\n\n\nResult of optimization:\n", result, "\n\n\nInvestigating stability of unoptimized NCS small delay closed loop system...\n\n\n")

            points1 = investigate_stability(A, B, K_static_Q2,
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            tau_range_function=lambda _h: (0.0, _h), 
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete) 

            print("\n\n\nInvestigating stability of optimized NCS small delay closed loop system...\n\n\n")

            points2 = investigate_stability(A, B, Mat([result.x]), 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            tau_range_function=lambda _h: (0.0, _h), 
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete)  

            display_multiple_h_tau_results([(points1[0], points1[1]), (points2[0], points2[1])],
                                           ["Stable $(h, \\tau)$-points Static Contr.", "Stable $(h, \\tau)$-points Dynamic Contr."],
                                           "Stability of ($h$, $\\tau$) Combinations for Fixed $0 \\leq \\tau < h$") 

        case 3:

            print("\n\n\nInvestigating stability of NCS small delay closed loop system...\n\n\n")

            points1 = investigate_stability(A, B, K_static_Q2, 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete, 
                                            tau_range_function=lambda _h: (0.0*_h, _h))  

            print("\n\n\nInvestigating stability of NCS large delay closed loop system...\n\n\n")

            points2 = investigate_stability(A, B, K_static_Q3, 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            matrix_creator_function=create_NCS_large_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete, 
                                            tau_range_function=lambda _h: (0.0*_h, 2.0*_h))  

            display_multiple_h_tau_results([(points1[0], points1[1]), (points2[0], points2[1])],
                                           ["Stable $(h, \\tau)$-points Static Contr. Small Delay", "Stable $(h, \\tau)$-points Static Contr. Large Delay"],
                                           "Stability of ($h$, $\\tau$) Combinations for Fixed $0 \\leq \\tau < 2.0h$") 

        case 4:

            print("\n\n\nOptimizing discrete model full state feedback for [0, tau_max] to be as large as possible for h = "+str(h_Q3_2)+".\n\n\n")

            result = optimize_discrete_feedback(K_static_Q3, -50, 50, max_stable_tau,
                                                h_Q3_2, A, B, K_static_Q3.shape,
                                                matrix_creator_function=create_NCS_large_delay_closed_loop_matrix,
                                                stability_checker_function=is_asymptotically_stable_discrete,
                                                tau_range_function=lambda _h: (0.0, 2.0*_h))

            print("\n\n\nResult of optimization:\n", result, "\n\n\nInvestigating stability of unoptimized NCS large delay closed loop system...\n\n\n")

            points1 = investigate_stability(A, B, K_static_Q3, 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            matrix_creator_function=create_NCS_large_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete,
                                            tau_range_function=lambda _h: (0.0, 2.0*_h))  

            print("\n\n\nInvestigating stability of optimized NCS large delay closed loop system...\n\n\n")

            points2 = investigate_stability(A, B, Mat([result.x]), 
                                            h_range=h_range, 
                                            h_steps=h_steps, 
                                            matrix_creator_function=create_NCS_large_delay_closed_loop_matrix, 
                                            stability_checker_function=is_asymptotically_stable_discrete,
                                            tau_range_function=lambda _h: (0.0, 2.0*_h))  

            display_multiple_h_tau_results([(points1[0], points1[1]), (points2[0], points2[1])],
                                           ["Stale $(h, \\tau)$-points Static Contr.", "Stable $(h, \\tau)$-points Dynamic Contr."],
                                           "Stability of ($h$, $\\tau$) Combinations for Fixed $0 \\leq \\tau < 2.0h$") 

        case 5:

            print("\n\n\nOptimizing discrete model full state feedback for $h_n$ to be as large as possible for combined controller.\n\n")

            result1 = optimize_discrete_feedback(K_static_Q4, -50, 50, max_stable_h_n,
                                                 h_n_min+0.0001, h_n_max, 1000, A, B, K_static_Q4.shape,
                                                 matrix_creator_function     = create_NCS_small_delay_closed_loop_matrix,
                                                 stability_checker_function  = is_asymptotically_stable_discrete)

            print("\n\n\nResult of optimization:\n", result1)
            print("\n\n\nOptimizing discrete model full state feedback for h_n to be as large as possible for sequence 1 controller.\n\n")

            result2 = optimize_discrete_feedback(K_static_Q4, -50, 50, max_stable_h_n,
                                                 h_n_min+0.0001, h_n_max, 1000, A, B, K_static_Q4.shape,
                                                 matrix_creator_function     = create_NCS_small_delay_closed_loop_matrix,
                                                 stability_checker_function  = is_asymptotically_stable_discrete,
                                                 include_two                 = False)

            print("\n\n\nResult of optimization:\n", result2)
            print("\n\n\nOptimizing discrete model full state feedback for h_n to be as large as possible for sequence 2 controller.\n\n")

            result3 = optimize_discrete_feedback(K_static_Q4, -50, 50, max_stable_h_n,
                                                 h_n_min+0.0001, h_n_max, 1000, A, B, K_static_Q4.shape,
                                                 matrix_creator_function     = create_NCS_small_delay_closed_loop_matrix,
                                                 stability_checker_function  = is_asymptotically_stable_discrete,
                                                 include_one                 = False)

            print("\n\n\nResult of optimization:\n", result3)
            print("\n\n\nInvestigating stability of resulting NCS small delay closed loop system for combined controller...\n\n\n")

            points1 = investigate_stability(A, B, Mat([result1.x]),
                                            h_range=h_range,
                                            h_steps=h_steps,
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix,
                                            stability_checker_function=is_asymptotically_stable_discrete,
                                            tau_range_function=lambda _h: (0.0, _h))

            print("\n\n\nInvestigating stability of resulting NCS small delay closed loop system for sequence 1 controller...\n\n\n")

            points2 = investigate_stability(A, B, Mat([result2.x]),
                                            h_range=h_range,
                                            h_steps=h_steps,
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix,
                                            stability_checker_function=is_asymptotically_stable_discrete,
                                            tau_range_function=lambda _h: (0.0, _h))

            print("\n\n\nInvestigating stability of resulting NCS small delay closed loop system for sequence 2 controller...\n\n\n")

            points3 = investigate_stability(A, B, Mat([result3.x]),
                                            h_range=h_range,
                                            h_steps=h_steps,
                                            matrix_creator_function=create_NCS_small_delay_closed_loop_matrix,
                                            stability_checker_function=is_asymptotically_stable_discrete,
                                            tau_range_function=lambda _h: (0.0, _h))


            display_multiple_h_tau_results([(points1[0], points1[1]), (points2[0], points2[1]), (points3[0], points3[1])],
                                           ["Stable points $(h, \\tau)$ comb. cntlr", "Stable points $(h, \\tau)$ seq. 1 cntlr", "Stable points $(h, \\tau)$ seq. 2 cntlr"],
                                           "Stability of ($h$, $\\tau$) Combinations for Fixed $0 \\leq \\tau < h$")

        case _:

            pass

    return 0






def welcome_user() -> int:
    print("\n -- What to run? --")
    print("""
          \n1: Question 2.1 (h, tau)-plot
          \n2: Question 2.2 K-optimization and resulting (h, tau)-plot
          \n3: Question 3.1 combined (h, tau)-plot
          \n4: Question 3.2 K-optimization and resulting (h, tau)-plot
          \n5: Question 4.3 K-optimizations and resulting combined (h, tau)-plots
          """)

    user_input = int(input(""))
    in_list = [1,2,3,4,5]
    while user_input not in in_list:
        user_input = int(input("Invalid input.\n\nValid inputs are: " + str(in_list)))

    return user_input
