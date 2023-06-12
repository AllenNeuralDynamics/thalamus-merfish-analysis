..\code\napari

This folder contains notebooks that require [napari](https://napari.org/stable/) to run.
These cannot be run from this capsule because they open a new window GUI to visualize & interact with images & figures, which is not a feature supported by Code Ocean at this time.

If you'd like to use this notebook, please download it and run it locally.

Note that napari installation (even via pip) can be finicky and dependent on your operating system version, as it relies on your OS's graphics software, which can vary. Reach out to Meghan if you run into errors trying to run notebooks in this folder after installing napari in your enviroment.


As an example, on a RockyLinux 7 machine, it was necessary to use a specific version of pyqt5.

Running:

    pip install napari[all] 

automatically installed the following package versions (2023-05-23):
- pyqt5=5.15.9
- pyqt5-qt5=5.15.2
- pyqt5-sip=12.12.1

It was then necessary to roll back the pyqt5 version to 5.14.0 like so:

    pip uninstall -y pyqt5 pyqt5-qt5 pyqt5-sip 
    pip install pyqt5==5.14.0 