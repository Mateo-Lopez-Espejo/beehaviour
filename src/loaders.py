from vame.util.auxiliary import read_config
from vame.analysis.community_analysis import get_labels, compute_transition_matrices

import my_paths as mp

# load transition matrices

def get_motif_series(data_names):
    cfg = read_config(mp.vame_path / "config.yaml")
    labels = get_labels(
        cfg=cfg,
        files=data_names,
        model_name=cfg['model_name'],
        n_cluster=cfg['n_cluster'],
        parameterization=cfg['parameterization']
    )

    return labels

def get_transition_matrices(data_names):
    cfg = read_config(mp.vame_path / "config.yaml")
    labels = get_labels(
        cfg=cfg,
        files=data_names,
        model_name=cfg['model_name'],
        n_cluster=cfg['n_cluster'],
        parameterization=cfg['parameterization']
    )


    # todo chekc if the output of this funciton is equal to that of mine
    #  besides the zero diagonal
    transition_matrices = compute_transition_matrices(
        data_names,
        labels,
        n_cluster=cfg['n_cluster'])

    return dict(zip(data_names, labels)), dict(
        zip(data_names, transition_matrices))

