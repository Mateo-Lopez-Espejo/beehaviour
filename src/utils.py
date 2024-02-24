import warnings

import numpy as np
import xarray as xr

from typing import List, Union


def get_transition_matrix(
        series: Union[np.array, List[np.array]],
        states: Union[None, List[str]] = None,
        keep_diagonal: bool = True
) -> xr.DataArray:
    """
    Stacks the series agains a shifted version of itself generating pairs
    of secuential points, i.e., transitions. then count them and set into a
    transition array
    Args:
        series:
        keep_diagonal: if False sets the diagonal of self transitions to zero

    Returns:

    """
    # single array to singleton list of arrays for consistent processing
    if type(series) == np.array:
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
        missing_deffined = set(observed_states).difference(set(states))
        if missing_deffined:
            warnings.warn(
                f"there are {missing_deffined} "
                f"in observed states not in set states"
            )
        missing_observed = set(states).difference(set(observed_states))
        if missing_observed:
            warnings.warn(
                f"there are {missing_observed} "
                f"in set states not in observed states"
            )
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
    trans_arr /= trans_arr.sum(dim="dest")

    if np.any(np.isnan(trans_arr)):
        trans_arr = np.nan_to_num(trans_arr)

    return trans_arr
