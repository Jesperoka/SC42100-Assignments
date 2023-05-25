from ..core.typedefs import *


# Assignment parameters
A               = Mat([[5 - 0, 0.5 - 8], 
                       [0    , 1      ]])

B               = Mat([[0.0],
                       [1.0]])


# Question 1 parameters:
K_bar_1         = Mat([[-16.0/3, 8.0]]) # solution to pole placement problem
K_bar_2         = Mat([[-28.0/5, 9.0]]) # solution to pole placement problem
K_bar_3         = Mat([[-8.0/5 , 3.0]]) # solution to pole placement problem


# Question 2 parameters:
h_Q2_2          = 0.25
K_static_Q2     = BlkMat2D([(K_bar_1, Zeros1x1)])
K_static_Q2_2   = BlkMat2D([(K_bar_2, Zeros1x1)])
K_dynamic_Q2    = Mat([[-3.874e+00, 7.033e+00, 2.226e-01]]) # result of optimization


# Question 3 parameters:
h_Q3_2          = 0.0777
K_static_Q3     = BlkMat2D([(K_bar_1, Zeros1x1, Zeros1x1)])
K_dynamic_Q3    = Mat([[-9.409e+00, 1.535e+01, 8.049e-01, 5.244e-01]]) # result of optimization

# Question 4 parameters
h_n_min         = 0.0
h_n_max         = 1.0
K_static_Q4     = BlkMat2D([(K_bar_1, Zeros1x1)])
K_combined_Q4   = Mat([[-6.685e+00, 1.200e+01, 9.955e-01]]) # result of optimization
#K_sequence_1_Q4 = Mat([[]])
