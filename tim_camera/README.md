# Object detection and distance calcualtion

This program detects objects and measures the distance between them in 3 dimensional space (e.g. humans). See the [documentation](https://sourceforge.net/p/vuong-aidem/code/ci/master/tree/tim_camera/Documentation/Dokumentation_Kamera_Abstand.pdf) (German) for further information.

Note: the project is still an early version. Code refactors are still made for better appearance and better understandable code.

## Use cases

This program can be used to observe sourroundings with a camera and measure the distance between objects, e.g. persons in a store or cars in a carpark.

## Installation:

This program is written in **python 3.6 +**. It is also optimized for usage with a [Raspberry Pi](https://www.raspberrypi.org/) (it is recommended to use a Raspberry Pi 4) but it should work for various devices with a camera.

1. Download or clone the repository
2. (*optional*) install python [virtualenv](https://virtualenv.pypa.io/en/stable/) and create an evironment
3. Install pip dependecies (numpy, tflite, opencv) with `pip3 install -r requirements.txt` or manually with
    * `pip3 install numpy`
    * `pip3 install opencv-python`
    * `pip3 install tflite`
    * `pip3 install tensorflow`

Instead of installing *tflite* and *tensorflow* one can install [*tflite runtime*](https://www.tensorflow.org/lite/guide/python)

Optionally you can install opencv on linux with "`sudo apt-get install libopencv-dev python-opencv`";

When everything is installed youa re ready to start!

## Usage

The program comes with a GUI written with Tkinter. To start the GUI execute the file `detection_webcam_gui.py`. You can choose between live detection and video detection.

Please note that the GUI is still WIP and might change its style and functionality over time.

### Live Detection

To start the live detection you need to press the corresponding button. Now you can start the detection, however for the calculations for the object distances it is necessary to calibrate the program. For the calculation of an object to the filming camera it is necessary to know the focal width. This can be done by selecting the option in the menu (only possible after selecting live detection). A second window will be opened where an object for calibration has to be specified (width, distance and name; Please note that the object should be one that can be detected. For a list of detectable objects see [the labelmap](https://sourceforge.net/p/vuong-aidem/code/ci/master/tree/tim_camera/Sample_Model/labelmap.txt)). Be aware that a change in the focal width will result in faulty calculations and a new calibration is needed.

### Video Detection

For the video detection you need to specify the path to the video and the focal width of the used camera to film the video. Please note that a change in the focal width within the video may result in faulty calculations for the distances.

### Options

The program offers a few options to set (*more will be added later*). Currently there are four options:

* Autosave: If the checkbox is ticked, the recorded videostream will be saved to a file. The filename will always be 'detection_YYYY_MM_DD_hh_mm_ss'.
* Detection objects: Only the objects specified here will be processed. Other detectable objects will still be detected but ignored. The input must be comma delimited.
* No detection objects: analogical to the detectiojn objects. The objects specified in this list will be ignored while all other will be processed. The input must be comma delimited.
* Video folder: The path to the standard video folder where videos will be loaded from and stored to.

*Info:* For all detectable objects see [the labelmap](https://sourceforge.net/p/vuong-aidem/code/ci/master/tree/tim_camera/Sample_Model/labelmap.txt).

**NOTE**: currently the settings will not be saved when restarting the program.

## Known Bugs

* If the camera is used by another program it can't be accessed by the detection program. To fix this you probably need to have a look at your computer settings.
* Autsave will save an empty video.

## Upcoming features

* Refactor frames to autoscale contents with frame size
* It is planned to implement a help page in the program for easier usage.
