import warnings

import numpy as np
import xarray as xr

from typing import List, Union


def get_transition_matrix(
        series: Union[np.ndarray, List[np.ndarray]],
        states: Union[None, List[str]] = None,
        keep_diagonal: bool = True,
        norm_by_row: bool = True
) -> xr.DataArray:
    """
    Stacks the series agains a shifted version of itself generating pairs
    of secuential points, i.e., transitions. then count them and set into a
    transition array
    Args:
        series:
        states: list of unique states expected to be observed in the series
        keep_diagonal: if False sets the diagonal of self transitions to zero
        norm_by_row: normalizes by rows so the sum == 1, i.e. transition probability
    Returns:

    """
    # single array to singleton list of arrays for consistent processing
    if type(series) == np.ndarray:
        series = [series]

    transitions = list()
    counts = list()
    observed_states = list()
    for ser in series:
        # unique with defined axis cannot handle 'Object' dtype
        # if ser.dtype == "O":
        ser = ser.astype('str')

        trns, cnt = np.unique(
            np.stack([ser[:-1], ser[1:]], axis=1),
            axis=0, return_counts=True
        )

        transitions.append(trns)
        counts.append(cnt)
        observed_states.extend(np.unique(ser).tolist())

    observed_states = np.sort(np.unique(observed_states)).tolist()
    if states:
        discrepant = set(observed_states).difference(set(states))
        if discrepant:
            msg = (
                f"states {discrepant} are discrepant beteen "
                   f"\nobserved {set(observed_states)} and "
                   f"\ndefined {set(states)}"
            )
            print(msg)
            # warnings.warn(msg)

    else:
        states = observed_states

    n_states = len(states)
    trans_arr = xr.DataArray(
        np.zeros([n_states, n_states]),
        dims=("src", "dest"), coords={"src": states, "dest": states}
    )

    # finally goes though the transition and counts of every file
    # and through the individuale transition and number
    for trns, cnt in zip(transitions, counts):
        for t, c in zip(trns, cnt):
            trans_arr.loc[t[0], t[1]] += c

    if not keep_diagonal:
        for st in states:
            trans_arr.loc[st, st] = 0

    # Normalizes by row sum to get the transition probability
    # automatic dimension alignment with xarray magic.
    if norm_by_row:
        trans_arr /= trans_arr.sum(dim="dest")

    if np.any(np.isnan(trans_arr)):
        trans_arr = xr.DataArray(
            np.nan_to_num(trans_arr),
            dims = trans_arr.dims,
            coords = trans_arr.coords
        )

    return trans_arr
