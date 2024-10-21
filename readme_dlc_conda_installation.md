# installing and making deeplabcut work is a PITA

1.  install cudatoolkit and cudnn of the right versions

```shell
conda install -c conda-forge tensorflow-gpu
```

2. deeplabcut says it depends on pyside, yet it installed pyqt last time, which was giving out errors.
furthermore, this might not be a necesary step as pyside is an alterlative to qtpy, installed below

```shell
conda install -c conda-forge pyside
```

3. if there is an issue with qt, on which pyside depends (?) install with  pip
```shell
pip install qtpy
```

### the conda installation is finally working. To install the exact same environment run:
The environment file should be on the same file as this readme and can be used
to recreate the environment. 

```shell
conda create --name myenv --file dlc_working_frozen.txt
```

