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
K_static_Q2     = BlkMat2D([(K_bar_1, Zeros1x1)])
K_dynamic_Q2    = Mat([[-3.012e+00, 5.517e+00, 0.0, 0.0]]) # result of optimization


# Question 3 parameters:
K_static_Q3     = BlkMat2D([(K_bar_1, Zeros1x1, Zeros1x1)])
#K_dynamic_Q3    = Mat([]) # result of optimization


# Question 4 parameters
