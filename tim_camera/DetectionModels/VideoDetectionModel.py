""""Specified Model for object detection with tensorflow and distance calculation."""

__version__ = "1.0.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz@stud.uni-frankfurt.de"
__credits__ = "Special thanks to The Anh Vuong who came up with the original idea." \
              "This code is also based off of the code from Evan Juras"

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

import cv2.cv2 as cv2
import numpy as np
import os

from aidem.tim_camera.DetectionModels.BasisModel import Detection, Functionality


class VideoDetection(Detection):
    """
        Class for video object detection
        """

    def __init__(self,
                 model_name: str = 'Sample_Model',
                 graph_name: str = 'ssd_mobilenet_v1_1_metadata_1.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.6,
                 debug: bool = False
                 ):
        """
        Video object detection and distance calculation.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param distance_threshold: minimum distance value between objects
        :param debug: Option to show debug information
        """

        super(VideoDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold)

        self._video = None
        self.stop = False

        # Initialise default value for objects width (in cm) and distance threshold (in cm)
        self.objects_width_cm = 50
        self.distance_threshold = 50

    def detect(self,
               video_name: str = "Sample_Video/testvideo1.mp4",
               focal_width: int = 1000,
               detect: list = [],
               no_detect: list = [],
               functionality: Functionality = Functionality.Detection,
               debug: bool = False
               ):
        """
        Object detection via a video feed and distance calculations of the objects.

        :param video_name: path to the video that should be used for detection
        :param focal_width: focal width of the used camera in the video
        """

        self.focal_value = focal_width

        # Path to video file
        video_path = os.path.join(self._cwd_path, video_name)

        color_variation = 0

        # Open video file
        self._video = cv2.VideoCapture(video_path)
        imW = self._video.get(cv2.CAP_PROP_FRAME_WIDTH)
        imH = self._video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while (self._video.isOpened()):

            objects_count = 0

            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = self._video.read()
            if not ret:
                print('Reached the end of the video!')
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._width, self._height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self._floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
            self._interpreter.invoke()

            # Retrieve detection results
            boxes = self._interpreter.get_tensor(self._output_details[0]['index'])[
                0]  # Bounding box coordinates of detected objects
            classes = self._interpreter.get_tensor(self._output_details[1]['index'])[
                0]  # Class index of detected objects
            scores = self._interpreter.get_tensor(self._output_details[2]['index'])[0]  # Confidence of detected objects

            coords = []

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0)):

                    object_name = self._labels[int(classes[i])]

                    # Filter
                    if (len(detect) > 0):
                        if object_name not in detect:
                            continue
                    if (len(no_detect) > 0):
                        if object_name in no_detect:
                            continue

                    objects_count += 1

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    if (i + 1) * 40 > 255:
                        color_variation += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (10, (40 + (40 * i)) % 255, (color_variation * 40) % 255), 2)

                    # Save coordinates of detected person
                    coords.append([[xmin, ymin], [xmax, ymax]])

                    if functionality == Functionality.Distance:
                        if (len(coords) > 1 and False):

                            if object_name == "person":
                                result = self._distance_calculation(coords, debug=debug)
                            else:
                                result = self._distance_calculation(coords, debug=debug)

                            if (debug):
                                print(result)

                            if result[0] == 3:
                                pass
                            elif result[0] == 0:
                                self._draw(frame, coords, result[1], result[2], result[3], result[4])
                            elif result[0] == 1:
                                self._draw(frame, coords, result[1], result[2], result[3], result[4])
                            elif result[0] == 2:
                                pass
                            else:
                                raise Exception("Invalid distance calculation result.")

                    # Demo label
                    cv2.putText(frame, 'DEMO', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            if functionality == Functionality.Counting:
                # Objects counting
                cv2.putText(frame, 'Objects on screen: {}'.format(objects_count), (10, int(self._imH*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2, cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

    def __del__(self):
        """
        Destructor.
        """

        # Clean up
        if self._video is not None:
            self._video.release()
        cv2.destroyAllWindows()