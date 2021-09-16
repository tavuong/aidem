# Object detection and distance calcualtion

This program implements a smart camera with object detection and deepfake functionalitys. The applications are

- Detection objects
- Counting objects in a frame
- Measuring the distance between objects in a frame
- FaceSwap in a Livestream

## Installation:

This program is written in **python 3.6 +**. It is also optimized for usage with a [Raspberry Pi](https://www.raspberrypi.org/) (it is recommended to use a Raspberry Pi 4) but it should work for various devices with a camera.

1. Download or clone the repository
2. (*optional, recommended*) install python [virtualenv](https://virtualenv.pypa.io/en/stable/) and create an evironment
3. Install pip dependecies (numpy, tflite, opencv) with `pip3 install -r requirements.txt` or manually with
    * `pip3 install numpy`
    * `pip3 install opencv-python`
    * `pip3 install tflite`
    * `pip3 install tensorflow`
    * `pip3 install dlib`

Instead of installing *tflite* and *tensorflow* one can install [*tflite runtime*](https://www.tensorflow.org/lite/guide/python) on devices it is available.

Optionally you can install opencv on linux with `sudo apt-get install libopencv-dev python-opencv`;

When everything is installed youa re ready to start!

## Usage

The program comes with a GUI written with Tkinter. To start the GUI execute the file `detection_webcam_gui.py`. You can choose between a live detection, a video detection and a FaceSwap.

### Live Detection

To start the live detection you need to press the corresponding button. Now you can start a standrad detection, object counting or distance measuring. For the distance measuring it is necessary to do a calibration first via the menu.

### Video Detection

For the video detection you need to specify the path to the video. Currently the video detection only supports object counting.

### Face swap

To start a face swap you need to select the LiveFaceSwap and then 

### Options

The program offers a few options to set (*more will be added later*). Currently there are four options:

* Autosave: If the checkbox is ticked, the recorded videostream will be saved to a file. The filename will always be 'detection_YYYY_MM_DD_hh_mm_ss'.
* Detection objects: Only the objects specified here will be processed. Other detectable objects will still be detected but ignored. The input must be comma delimited.
* No detection objects: analogical to the detectiojn objects. The objects specified in this list will be ignored while all other will be processed. The input must be comma delimited.
* Video folder: The path to the standard video folder where videos will be loaded from and stored to.

**NOTE**: currently the settings will not be saved when restarting the program.

## Known Bugs

* If the camera is used by another program it can't be accessed by the detection program.
* Autsave will save an empty video.
