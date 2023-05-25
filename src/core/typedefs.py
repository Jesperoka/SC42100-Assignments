"""File for creating type aliases for brevity"""
from collections.abc import Callable
from numpy import array, block, float64, csingle, ndarray, concatenate
from typing import Final
from functools import partial
from numba import jit, typeof

# Type aliases
number_type = csingle
fn          = Callable
Mat         = partial(array , dtype=number_type)
BlkMat      = block

@jit(nopython=True)
def BlkMat2D(list_of_row_of_matrices: list[tuple[ndarray]]): # TODO: change to tuple of tuples just to remove annoying deprecation warning
    block_matrix = concatenate(list_of_row_of_matrices[0], axis=1)
    for row_of_matrices in list_of_row_of_matrices[1:]:
        matrix_row = concatenate(row_of_matrices, axis=1)
        block_matrix = concatenate((block_matrix, matrix_row), axis=0)
    return block_matrix

# Constant matrices
I1x1:     Final[Mat] = Mat([[1.0]])

I2x2:     Final[Mat] = Mat([[1.0, 0.0],
                            [0.0, 1.0]])

Zeros1x1: Final[Mat] = Mat([[0.0]])

Zeros1x2: Final[Mat] = Mat([[0.0, 0.0]])

Zeros2x1: Final[Mat] = Mat([[0.0],
                            [0.0]])

Zeros2x2: Final[Mat] = Mat([[0.0, 0.0],
                            [0.0, 0.0]])
