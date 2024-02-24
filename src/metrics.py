import numpy as np
from pandas import Series

from src.utils import get_transition_matrix

def get_flower_constancy_index(series: Series):
    """
    Flower constancy index as defined in Waser 1986
    Args:
        series:

    Returns:

    """

    trans_mat = get_transition_matrix(np.asarray(series))
    # todo generalize to n * n matrices, probably geting the diagonal and the upper and lower triangles
    const = float(
        (np.sqrt(trans_mat[0, 0] * trans_mat[1, 1]) -
         np.sqrt(trans_mat[0, 1] * trans_mat[1, 0]))
        /
        (np.sqrt(trans_mat[0, 0] * trans_mat[1, 1]) +
         np.sqrt(trans_mat[0, 1] * trans_mat[1, 0]))
    )
    if np.isnan(const):
        # edge case where there is only visits to one type of flowers, i.e., no transitions
        const = 1.0

    return const