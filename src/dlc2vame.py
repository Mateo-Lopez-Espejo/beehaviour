from pathlib import Path
import warnings
from typing import Tuple, List


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from vame.util.auxiliary import read_config, write_config

def _validate_paths(df_source: Path, vid_source: Path, dest: Path):
    # chekc for valid data frame files
    df_files = list(df_source.glob("*_filtered.h5"))
    if df_files:
        print(f"found {len(df_files)} filtered pose .h5 files at {df_source}")
    else:
        raise FileNotFoundError(
            f"found no filtered pose .h5 files at {df_source}")

    # check for valid video files
    vid_files = list(vid_source.glob("*.mp4"))
    if vid_files:
        print(f"found {len(vid_files)} mp4 files at {vid_source}")
    else:
        raise FileNotFoundError(
            f"found no mp4 files at {vid_source}")

    # check for the bijection between dataframes and videos, warn if not.
    df_names = {f.stem.split("DLC")[0] for f in df_files}
    vid_names = {v.stem for v in vid_files}

    if df_names.difference(vid_names):
        warnings.warn(
            f"Found pose dataframes without corresponding videos:\n"
            f"{df_names.difference(vid_names)}"
        )
    elif vid_names.difference(df_names):
        warnings.warn(
            f"Found videos without corresponding pose dataframes:\n"
            f"{vid_names.difference(df_names)}"
        )
    else:
        print("every video has a corresponding pose estimation. Great!")

    if not dest.exists():
        raise FileNotFoundError("The destination folder does not exist."
                                " Create one with the vame utility")

    comon = df_names.intersection(vid_names)
    out_df_files = [f for f in df_files if f.stem.split("DLC")[0] in comon]
    out_vid_files = [v for v in vid_files if v.stem in comon]

    return out_df_files, out_vid_files


def _drop_occluded_body_parts(df: pd.DataFrame,
                              confidence:float) -> pd.DataFrame:
    # drops body parts that have no likely points, i.e., which are occluded.
    bad_parts = list()
    for bodypart in df.columns.get_level_values('bodyparts').unique():
        likes = df.loc[:, (slice(None), bodypart, "likelihood")].values
        if np.all(likes <= confidence):
            bad_parts.append(bodypart)
    print(f"{bad_parts} have no likely points, dropping from dataframe")
    df.drop(columns=bad_parts, level='bodyparts', inplace=True)

    return df


def _drop_empty_timepoints(df: pd.DataFrame, confidence:float, n_conf_labels=2):
    print('removing time stamps from df with no confident body parts...')
    enough_body_parts = np.sum(
        df.xs(axis=1, key="likelihood", level="coords") > confidence,
        axis=1
    ) >= n_conf_labels

    start_idx = np.where(enough_body_parts)[0][0]
    end_idx = np.where(enough_body_parts)[0][-1]
    print(
        f"...kept {end_idx + 1 - start_idx} time points out of {df.shape[0]}!"
    )
    df = df.iloc[start_idx:end_idx + 1, :]

    plt.plot(enough_body_parts)
    plt.vlines([start_idx, end_idx + 1], ymin=0, ymax=1, color="red")
    plt.show()

    return df, (start_idx, end_idx)


def _copy_cropped_video(source_file: Path, dest_file: Path, start_frame: int,
                        end_frame: int):
    print("removing empty frames ...")
    cap = cv.VideoCapture(str(source_file))
    ret, frame = cap.read()
    h, w, _ = frame.shape
    fps = cap.get(cv.CAP_PROP_FPS)

    sliced_video_file = dest_file

    fourcc = cv.VideoWriter_fourcc(*"avc1")
    writer = cv.VideoWriter(str(sliced_video_file), fourcc, fps, (w, h))

    f0 = 0
    f1 = 0
    while ret:
        f0 += 1
        if start_frame <= f0 <= end_frame:
            f1 += 1
            writer.write(frame)
        ret, frame = cap.read()

    print(f"...sliced {f1} out of {f0} frames")

    writer.release()
    cap.release()

    return dest_file

    return df, [start_idx, end_idx]


def _get_dataset_bodyparts(vame_path: Path) -> dict:
    # todo make it handle datasets located not in the vame folder

    # read the already present files for old comon body parts
    datum_bodyparts_dict = dict()
    for existing_df_file in (vame_path / 'videos' / 'pose_estimation').glob(
            "*.h5"):
        datum_bodyparts_dict[existing_df_file.stem] = set(
            pd.read_hdf(existing_df_file
                        ).columns.get_level_values('bodyparts').unique()
        )
    return datum_bodyparts_dict


def _keep_comon_body_parts(comon_parts: set, vame_path: Path) -> None:
    for ff in (vame_path / 'videos' / 'pose_estimation').glob("*.h5"):
        df = pd.read_hdf(ff)

        # error case in which a file has missing parts, perhaps previously
        # removed
        missing_parts = comon_parts.difference(
            df.columns.get_level_values("bodyparts")
        )
        if missing_parts:
            raise ValueError(f"{ff.stem} is missing {missing_parts}")

        parts_to_drop = set(
            df.columns.get_level_values("bodyparts")
        ).difference(comon_parts)

        if parts_to_drop:
            print(f"dropping {parts_to_drop} from {ff.stem} df")
            df.drop(columns=parts_to_drop, level='bodyparts', inplace=True)

            # overwrites original file!
            df.to_hdf(ff, '/df_with_missing')
        else:
            print(f"no parts to drop from {ff.stem} df, skipping")
            continue
    return None


def _remove_data(data_stems: List['str'], vame_path: Path) -> None:
    for fstem in data_stems:
        if (vame_path / 'videos' / fstem).with_suffix(".mp4").exists():
            print(f"deleting {fstem} mp4 and h5 files")
            _ = (vame_path / 'videos' / fstem).with_suffix(".mp4").unlink()
            _ = (vame_path / 'videos' / 'pose_estimation' / fstem).with_suffix(
                ".h5").unlink()

    return None


def _plot_common_parts(vame_path: Path):
    parts_dict = _get_dataset_bodyparts(vame_path)

    comon_parts = set.intersection(*parts_dict.values())
    all_parts = set.union(*parts_dict.values())

    print(f"all body parts {all_parts}"
          f"\ncomon body parts to all recording: \n{comon_parts}"
          f"\n {len(comon_parts)}")

    comon_grid = pd.DataFrame(columns=all_parts, index=parts_dict.keys(), )

    for key, val in parts_dict.items():
        for v in val:
            comon_grid.loc[key, v] = True

    comon_grid.fillna(value=False, inplace=True)

    print(comon_grid)

    # x -> body parts, y -> recordings
    plt.imshow(comon_grid, origin='upper')
    plt.xticks(ticks=range(len(comon_grid.columns)),
               labels=comon_grid.columns.tolist(),
               rotation=90)
    plt.yticks(ticks=range(len(comon_grid.index)),
               labels=comon_grid.index.tolist())
    plt.show()

    return plt.get_current_fig_manager()


def add_data_to_vame(df_source: Path, vid_source: Path, vame_path: Path,
                     exclude: Tuple['str']=(), reload: bool = False) -> None:

    # ready paths and generate config
    df_files, vid_files = _validate_paths(df_source, vid_source, vame_path)
    cfg = read_config(vame_path / "config.yaml")
    pose_confidence = cfg['pose_confidence']

    # since we might need to limit VAME training to body parts present
    # on all pose estimations, keep a list of the present body parts of
    # each file for later further processing

    if reload:
        old_data_body_parts = dict()
    else:
        old_data_body_parts = _get_dataset_bodyparts(vame_path)

    # preprocess new files to add to the dataset and keeps track of their
    # comon body parts
    new_data_body_parts = dict()
    for df_source_file in df_files:
        # dataframe preprocessing and copying
        datum_name = df_source_file.stem.split('DLC')[0]
        df_dest_file = (
                vame_path / 'videos' / 'pose_estimation' / datum_name
        ).with_suffix('.h5')

        if df_dest_file.exists():
            if reload:
                pass
            else:
                print(f"{datum_name}.h5 already transferred, skipping")
                continue
        elif datum_name in exclude:
            print(f"{datum_name}.h5 excluded, skipping")
            continue

        print(f"\nreading df from {df_source_file} and preprocessing...")
        df = pd.read_hdf(df_source_file)
        df = _drop_occluded_body_parts(df, pose_confidence)
        df, (start_frame, end_frame) = _drop_empty_timepoints(df, pose_confidence)

        new_data_body_parts[datum_name] = set(
            df.columns.get_level_values('bodyparts').unique()
        )

        # saves the preprocessed df
        df.to_hdf(df_dest_file, '/df_with_missing')
        print(f"... saved preprocessed df to {df_dest_file}!")

        # copies over the video, cutting to match kept pose estimation frames
        vid_source_file = (vid_source / datum_name).with_suffix(".mp4")
        vid_dest_file = (vame_path / "videos" / datum_name).with_suffix(".mp4")
        print(f"reading video from {vid_source_file} and time-cropign ...")

        _ = _copy_cropped_video(
            vid_source_file, vid_dest_file, start_frame, end_frame
        )

        print(f"... saved cropped video to {vid_dest_file}!\n")

    # compares old comon to new comon body parts
    # all_parts = set.union(*new_comon_dict.values())

    if new_data_body_parts:
        lost_parts = set.intersection(*old_data_body_parts.values()).difference(
            set.intersection(*new_data_body_parts.values())
        )
        response = 'dummy'
        while response not in ['', 'y', 'n', 'yes', 'no']:
            response = input(
                f"adding the new file(s):\n{list(new_data_body_parts.keys())}\n"
                f"will remove the common body part(s):\n{lost_parts}\n"
                f"do you wish to continue adding? y, n"
            ).lower()

        if response in ['y', 'yes']:
            data_body_parts = {**old_data_body_parts, **new_data_body_parts}
            print("\nKeeping new data. and recalculating comon body parts")

        elif response in ['n', 'no']:
            print(
                "\nRemoving newly added data and leaving comon body parts as is"
            )
            data_body_parts = old_data_body_parts
            _ = _remove_data(list(new_data_body_parts.keys()), vame_path)

        elif response == '':
            print("\nSkipping step matching body parts across data")
            return None

    else:
        print("\nNo new data to add")
        data_body_parts = old_data_body_parts

    comon_parts = set.intersection(*data_body_parts.values())
    _ = _keep_comon_body_parts(comon_parts, vame_path)


    # advice edition on config file and downstream function use
    df = pd.read_hdf(df_dest_file)
    ordere_bodyparts = df.columns.get_level_values(
        "bodyparts").unique().tolist()

    print(f"{len(ordere_bodyparts)} total body parts\n"
          f"set 'num_features: {len(ordere_bodyparts) * 2}' in the config.yaml file\n"
          f"and the right pose_ref_index in the indices when calling "
          f"vame.egocentric_alignment")

    for idx, bodypart in enumerate(ordere_bodyparts):
        print(f"{idx}: {bodypart}")

    # writes new files to yaml
    print("adding data names to yaml file, this destroys the anotations in the file!")
    cfg['video_sets'] = list(data_body_parts.keys())
    cfg['num_features'] = int(len(ordere_bodyparts) * 2)
    _ = write_config(vame_path / "config.yaml", cfg)

    return None