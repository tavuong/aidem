"""Basis Model for object detection with tensorflow and distance calculation."""

__version__ = "1.0.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz:stud.uni-frankfurt.de"
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
import sys
import time
import importlib.util
import math


class Detection:
    """
    Class for detection models basis
    """

    def __init__(self,
                 model_name: str = 'Sample_Model',
                 graph_name: str = 'detect.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.5,
                 use_tpu: str = '',
                 distance_threshold: int = 150,
                 debug: bool = False
                 ):
        """
        Basis model for video detection and object distance calculations.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param use_tpu: specifier if a TPU is to be used
        :param distance_threshold: minimum distance value between objects
        """

        self._min_conf_threshold = min_conf_threshold
        self.distance_threshold = distance_threshold

        self.do_detect = True

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_tpu:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_tpu:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_tpu:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (graph_name == 'detect.tflite'):
                graph_name = 'edgetpu.tflite'

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
        # If using Edge TPU, use special load_delegate argument
        if use_tpu:
            self._interpreter = Interpreter(model_path=path_to_ckpt,
                                            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(path_to_ckpt)
        else:
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
        self.focal_value = 0

    def _distance_calculation(self,
                              object_coordinates: list = [],
                              avg_width: int = 53,
                              debug: bool = False
                              ):
        """
        Distance calculation of detected objects.

        :param object_coordinates: 2 dimensional matrix of coordinates for the objects
        :param avg_width: (estimated) wisth of the detection objects
        :return: list of distance information (positioning of the objects, second object,
        shortest distance of objects, distance threshold in pixel)
        """

        # Initialize lists for various measurements
        proportion_x = []
        proportion_y = []
        camera_distance = []

        # Initialise proportion lists for later easy acces
        for i in range(len(object_coordinates)):
            proportion_x.append(0)
            proportion_y.append(0)

        for j in range(len(object_coordinates)):
            # Measure height and width of detected person (in pixel)
            proportion_x[j] = object_coordinates[j][1][0] - object_coordinates[j][0][0]
            camera_distance.append((self.focal_value * avg_width) / proportion_x[j])

            one_pixel = proportion_x[j] / avg_width
            if (debug):
                print("Length of one pixel in cm:" + str(one_pixel))
                print("Object " + str(j) + " - Distance to camera:", camera_distance[-1])

            if (j > 0):
                min_dist_pixels = one_pixel * self.distance_threshold

                for k in range(j):
                    # Horizontal distance of the detected objects
                    min_dist_obj_x_1 = abs(object_coordinates[j][1][0] - object_coordinates[k][0][0])
                    min_dist_obj_x_2 = abs(object_coordinates[k][1][0] - object_coordinates[j][0][0])

                    # Distance objects on z axis
                    dist_obj_z = abs(camera_distance[j] - camera_distance[k])

                    if (debug):
                        print("Object " + str(j) + ", " + str(k) + " - Distance to camera:", camera_distance[-1])

                    # Check for shortest distance between the objects
                    if (min_dist_obj_x_1 < min_dist_obj_x_2):
                        objects_distance = math.sqrt(min_dist_obj_x_1 ** 2 + dist_obj_z ** 2)
                        case = 0
                    elif (min_dist_obj_x_2 < min_dist_obj_x_1):
                        objects_distance = math.sqrt(min_dist_obj_x_2 ** 2 + dist_obj_z ** 2)
                        case = 1
                    else:
                        objects_distance = 0
                        case = 2

                    # Check if the shortest distance between the objects is smaller than the threshold
                    if (objects_distance < min_dist_pixels):
                        return [case, j, k, objects_distance, min_dist_pixels]
                    else:
                        return [3, j, k, objects_distance, min_dist_pixels]

        return [3]

    def _draw(self,
              frame,
              object_coordinates: list,
              object_position_1: int,
              object_position_2: int,
              objects_distance: int,
              min_dist: int,
              debug: bool = False
              ):
        """
        :param frame: video frame to paint on
        :param object_coordinates: 2 dimensional matrix of the object coordinates
        :param object_position_1: position of the first objects coordinates in the matrix
        :param object_position_2: position of the second objects coordinates in the matrix
        :param objects_distance: distance between the two objects
        :param min_dist: distance threshold of the objects (in pixel)

        """

        # Draw distance line
        cv2.line(frame, (object_coordinates[object_position_1][1][0], object_coordinates[object_position_1][1][1]),
                 (object_coordinates[object_position_2][0][0], object_coordinates[object_position_2][1][1]),
                 (255, 10, 0), 2)

        # Draw info box
        dist_label = '%s / %d' % (round(objects_distance, 2), round(min_dist, 2))
        dist_label_size, dist_base_line = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        dist_label_ymin = max(object_coordinates[object_position_1][1][1], dist_label_size[1] + 10)
        cv2.rectangle(frame, (object_coordinates[object_position_1][1][0], dist_label_ymin - dist_label_size[1] - 10),
                      (object_coordinates[object_position_1][1][0] + dist_label_size[0],
                       dist_label_ymin + dist_base_line - 10),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, dist_label, (object_coordinates[object_position_1][1][0], dist_label_ymin - 7),
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
