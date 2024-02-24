import pathlib as pl
import deeplabcut as dlc
import matplotlib
matplotlib.use('Agg')


# batch one, recorded wit a tripod, and a motion detection software
# videofolder = pl.Path("/home/mateo/motion/pilot_videos") # new video source
# config_path = "/home/mateo/code/beehaviour/data/beehaviour-Mateo-2023-09-14/config.yaml"


# batch two. First recording with the first generation of puzze flowers
# broken, pointing to a no longer existing folder with left_bend_01. (short)
# videofolder = pl.Path("/home/mateo/motion/curated_videos_plus")
# config_path = "/home/mateo/code/beehaviour/data/bottom_view-Mateo-2024-02-18/config.yaml"


# batch three. Second recording with the first generation of puzze flowers
# left and right zig_02.
videofolder = pl.Path("/home/mateo/motion/zig_02/curated")
config_path = "/home/mateo/code/beehaviour/data/bottom_view-Mateo-2024-02-20/config.yaml"





videos = [str(pp) for pp in videofolder.glob("*.mp4")]

# create project, annotated out since it's done, and I don't want to repeat it
config_path = dlc.create_new_project('bottom_view_DLC', 'Mateo',
                       videos, working_directory='/home/mateo/code/beehaviour/data',
                       copy_videos=False, multianimal=False)


# todo smarte new viedeo adition. What videos are already in the destination
# folder?
# dlc.add_new_videos(
#     config_path, videos, copy_videos=False
# )

# I figured, the easiest way is to make it automatic and then remove extra useless frames
dlc.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False)
# dlc.extract_frames(config_path, mode='manual')

# launches gui to add labels
dlc.label_frames(config_path)

# builds skeleton by drawing lines
dlc.SkeletonBuilder(
    config_path
)

# make images of selected frames with defined labels
dlc.check_labels(config_path)

# Packs the labeled data, applies data agumentation and splits train test
dlc.create_training_dataset(config_path, augmenter_type='imgaug')

# ensure to modify the pose_cfg.yml to use adam and batches >1 but not too big as to run out of memory
dlc.train_network(config_path, maxiters=500000, gputouse=0)


# todo set a sensible output path since this is the data that goes to VAME
output_path = pl.Path("/home/mateo/code/beehaviour/data/bottom_view_inference")

dlc.analyze_videos(
    config_path,
    videos,
    gputouse=0, save_as_csv=False,
    destfolder=str(output_path),
    dynamic=(False, .5, 10)
)

# todo figure out is sarima filterring is a better option and how to use it.
# remove low probability points
dlc.filterpredictions(
    config_path, video=videos,
    destfolder=str(output_path),
    save_as_csv=False
)

# check filtering
dlc.plot_trajectories(
    config_path, videos,
    filtered=True,
    destfolder=str(output_path),
    )


# todo, this function fails with some videos that have unspecified FPS.
#  Make a script to compress some of these videos to mp4 so this function behaves
# create a video containing labels, based on the previously labeled datasets
dlc.create_labeled_video(
    config=config_path,
    videos=videos,
    save_frames=False, trailpoints=3,
    draw_skeleton=True, skeleton_color='black',
    filtered=True,
    destfolder=str(output_path),
)