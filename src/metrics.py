from typing import Union, List

import numpy as np
from pandas import Series

from src.utils import get_transition_matrix

def bateman_flower_constancy_index(
        series: Series,
        states: Union[None, List[str]] = None,
        print_trans_mat: bool = False
):
    """
    Flower constancy index as defined in Waser 1986
    Args:
        series: series of strings
        states: list of strings representing unique states found in series
        print_trans_mat: bool, display the calculated transition matrix
    Returns:
        const: float, flower constancy index
    """

    trans_mat = get_transition_matrix(np.asarray(series), states=states, norm_by_row=True)
    if print_trans_mat:
        print(trans_mat.values)

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

def mateo_flower_constancy_index(
        series: Series,
        states: Union[None, List[str]] = None,
        print_trans_mat: bool = False
):
    """
    Home made flower constancy index defined by Mateo to solve issues wiht
    the geometric mean of zero values in the Bateman index.
    Args:
        series: series of strings
        states: list of strings representing unique states found in series
        print_trans_mat: bool, display the calculated transition matrix
    Returns:
        const: float, flower constancy index
    """

    trans_mat = get_transition_matrix(np.asarray(series), states=states, norm_by_row=False)
    if print_trans_mat:
        print(trans_mat.values)

    # todo generalize to n * n matrices, probably geting the diagonal and the upper and lower triangles
    same_trans = np.diagonal(trans_mat)

    const = float(np.sum(same_trans) / np.sum(trans_mat))

    return const