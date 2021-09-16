"""Basis Model for object detection with tensorflow and distance calculation."""

__version__ = "2.0.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz@stud.uni-frankfurt.de"
__credits__ = "Special thanks to The Anh Vuong who came up with the original idea." \
              "This code is also based off of code from Evan Juras"

# Description:
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video and calculates the distance between each of these objects.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# The code is also based off a raspberry pi tutorial for object detection:
# https://tutorials-raspberrypi.de/raspberry-pi-objekterkennung-mittels-tensorflow-und-kamera/
#

import os
import cv2.cv2 as cv2
import numpy as np
import importlib.util
import math
import enum
from threading import Thread


class Functionality(enum.Enum):
    Detection = 1
    Counting = 2
    Distance = 3


class VideoStream:
    """Camera object that controls video streaming from the camera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize camera and stream
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        self.width = resolution[0]

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.paused = False
        self.release_stream = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.release_stream:
                # Close camera resources
                self.stream.release()
                return
            elif not self.paused:
                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def read_ret(self):
        # Return frame and ret value
        return self.frame, self.grabbed

    def pause_stream(self):
        # Indicate that the stream should be stopped
        self.paused = True

    def continue_stream(self):
        # Indicate that stream should resume
        self.paused = False

    def stop_stream(self):
        # Indicate that stream should be released
        self.release_stream = True


class Detection:
    """
    Class for detection models basis
    """

    def __init__(self,
                 model_name: str = 'Sample_Model',
                 graph_name: str = 'ssd_mobilenet_v1_1_metadata_1.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.5,
                 debug: bool = False
                 ):
        """
        Basis model for video detection and object distance calculations.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param debug: Option to show debug information
        """

        self._min_conf_threshold = min_conf_threshold

        self.do_detect = True

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter

        # Get path to current working directory
        self._cwd_path = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        path_to_ckpt = os.path.join(self._cwd_path, model_name, graph_name)

        # Path to label map file
        path_to_labels = os.path.join(self._cwd_path, model_name, labelmap_name)

        # Load the label map
        with open(path_to_labels, 'r') as f:
            self._labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self._labels[0] == '???':
            del (self._labels[0])

        # Load the Tensorflow Lite model.
        self._interpreter = Interpreter(model_path=path_to_ckpt)
        self._interpreter.allocate_tensors()

        # Get model details
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._height = self._input_details[0]['shape'][1]
        self._width = self._input_details[0]['shape'][2]

        self._floating_model = (self._input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

        # Variable to hold focal width of used camera lens
        # Focal value is in unit pixels!
        self.focal_value = 0

    def _euclidean_distance_3D(self, vector1: list, vector2: list):
        return math.sqrt((vector2[0] - vector1[0])**2 + (vector2[1] - vector1[1])**2 + (vector2[2] - vector1[2])**2)

    def _distance_calculation(self,
                              object_coordinates: list = [],
                              center_coordinates: list = [],
                              distance_threshold: int = 50,
                              width_of_obj: int = 50,
                              debug: bool = False
                              ):
        """
        Distance calculation of detected objects.

        :param object_coordinates: 2 dimensional matrix of coordinates for the objects bounding boxes
        :param center_coordinates: 2 dimensional matrix containing the center coordinates for each object
        :param width_of_obj: (estimated) width of the detection objects
        :return: list of distance information (positioning of the objects, second object,
        shortest distance of objects, distance threshold in pixel)
        """

        # Initialize lists for various measurements
        proportion_x = []
        proportion_y = []
        camera_distance = []

        # Initialise proportion lists for later easy access
        for i in range(len(object_coordinates)):
            proportion_x.append(0)
            proportion_y.append(0)

        distance_results = []

        for j in range(len(object_coordinates)):
            # Measure height and width of detected person (in pixel)
            proportion_x[j] = abs(object_coordinates[j][1][0] - object_coordinates[j][0][0])
            # Ratio one pixel / one centimeter
            one_pixel = proportion_x[j] / width_of_obj

            # Distance of the object to the camera (in cm)
            camera_distance.append((self.focal_value * width_of_obj) / proportion_x[j])

            if (debug):
                print("Ratio pixel / cm:" + str(one_pixel))
                print("Object " + str(j) + " - Distance to camera (cm):", camera_distance[j])

            if (j > 0):
                # Min distance in pixel
                min_dist_pixels = one_pixel * distance_threshold

                for k in range(j):
                    # 3D coordinates (convert distance to camera to pixel here)
                    object_3D_coords_1 = [center_coordinates[j][0], center_coordinates[j][1],
                                          camera_distance[j]*one_pixel]
                    object_3D_coords_2 = [center_coordinates[k][0], center_coordinates[k][1],
                                          camera_distance[k]*one_pixel]

                    # Distance between the objects (in pixel)
                    distance_between_objects = self._euclidean_distance_3D(object_3D_coords_1, object_3D_coords_2)

                    # Check if distance is too short
                    if distance_between_objects < min_dist_pixels:
                        # The distance between the objects is too short
                        distance_in_cm = distance_between_objects / one_pixel
                        # Return result information
                        distance_results.append([j, k, distance_between_objects, distance_in_cm, center_coordinates])
                    else:
                        # The distance is not to short
                        pass

        return distance_results

    def _draw(self,
              frame,
              coordinates1: tuple,
              coordinates2: tuple,
              objects_distance_cm: int,
              min_dist_cm: int,
              debug: bool = False
              ):
        """
        :param frame: video frame to paint on
        :param coordinates1: Coordinates of the first object
        :param coordinates2: Coordinates of the second object
        :param objects_distance: distance between the two objects
        :param min_dist: distance threshold of the objects (in pixel)

        """

        # Draw distance line
        cv2.line(frame, coordinates1, coordinates2, (255, 10, 0), 2)

        # Draw info box
        dist_label = '%s / %d' % (round(objects_distance_cm, 2), round(min_dist_cm, 2))
        dist_label_size, dist_base_line = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        dist_label_ymin = max(coordinates1[1], dist_label_size[1] + 10)
        cv2.rectangle(frame, (coordinates1[0], dist_label_ymin - dist_label_size[1] - 10),
                      (coordinates1[0] + dist_label_size[0],
                       dist_label_ymin + dist_base_line - 10),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, dist_label, (coordinates1[0], dist_label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


"""
SANDBOX

# Average parameters
        self.avg_width_person = 45 + 8 + 4  # +8 due to borders not aligning to body
        self.avg_height_person = 172
        self.avg_proportion_person = self.avg_width_person / self.avg_height_person

        self.test_distance = 216

        # Old value:
        self.fokal_empir = 1500

        # Variable for new calibrated value:
        self.focal_value = 0
"""
