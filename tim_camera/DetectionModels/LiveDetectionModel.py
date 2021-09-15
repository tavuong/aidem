"""Specified Model for object detection with tensorflow and distance calculation."""

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
import time
import glob

from aidem.tim_camera.DetectionModels.BasisModel import VideoStream, Detection, Functionality


class LiveDetection(Detection):
    """
    Class for live object detection
    """

    def __init__(self,
                 model_name: str = 'Sample_Model',
                 graph_name: str = 'ssd_mobilenet_v1_1_metadata_1.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.5,
                 resolution: str = '1280x720',
                 debug: bool = False
                 ):
        """
        Live object detection and distance calculation.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param resolution: desired video resolution
        :param debug: Option to show debug information
        """

        super(LiveDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold)

        resW, resH = resolution.split('x')
        self._imW, self._imH = int(resW), int(resH)

        # Initialize frame rate calculation
        self._frame_rate_calc = 1
        self._freq = cv2.getTickFrequency()

        # Initialize video stream
        self._videostream = VideoStream(resolution=(self._imW, self._imH), framerate=30).start()
        time.sleep(1)

        self.undistortion_matrix = None

        # Initialise default value for objects width (in cm) and distance threshold (in cm)
        self.objects_width_cm = 50
        self.distance_threshold = 50

    def calibrate(self,
                  obj_width_cm: int = 0,
                  obj_dist_cm: int = 0,
                  obj_name: str = "",
                  sample_count: int = 10,
                  debug: bool = False
                  ):
        """
        Calculation for the focal width of used camera. Note that a change in the focal width after calibrating
        will result in faulty distance calculations.

        :param obj_width_cm: the width of the object to calibrate with
        :param obj_dist_cm: the distance of the object to calibrate with
        :param obj_name: the name of the object to calibrate with
        :param sample_count: number of samples to take for focal width calculation (mean will be used)
        :param debug: Toggle debug information

        :return: True as signal for GUI

        """

        color_variation = 0
        foc_measures = 0

        for i in range(sample_count):
            # Grab frame from video stream
            frame1 = self._videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
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
            scores = self._interpreter.get_tensor(self._output_details[2]['index'])[
                0]  # Confidence of detected objects

            obj_type = []

            for i in range(len(scores)):
                if (scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0):

                    # Check for the right object (ensure correct measurement when several objects are detected)
                    if self._labels[int(classes[i])] != obj_name:
                        continue
                    else:
                        obj_type.append(str(self._labels[int(classes[i])]))

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * self._imH)))
                    xmin = int(max(1, (boxes[i][1] * self._imW)))
                    ymax = int(min(self._imH, (boxes[i][2] * self._imH)))
                    xmax = int(min(self._imW, (boxes[i][3] * self._imW)))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (10, (40 + (40 * i)) % 255, (color_variation * 40) % 255), 2)

                    # Calculate object width in pixel
                    obj_width_pixels = xmax - xmin
                    # Sum up the focal lengths of all shots
                    foc_measures += (obj_width_pixels * obj_dist_cm) / obj_width_cm

                    # Draw label
                    object_name = self._labels[
                        int(classes[i])]  # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame, 'FPS: {0:.2f}'.format(self._frame_rate_calc), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2, cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

        self.focal_value = foc_measures / sample_count
        if (debug):
            print("Focal value:", self.focal_value)

        return True

    def calibrate_board(self,
                        cols: int = 0,
                        rows: int = 0,
                        debug: bool = False
                        ):
        """Calibration via chessboard with opencv

        :param cols: columns of the chessborad to be detected
        :param rows: rows of the chessboard to be detected
        :param debug: debug mode

        :return: if calibration was successful
        """
        debug = True

        img_counter = 0
        img_name = ""

        print("++ Press SPACE to take a photo of the chess board. Press ESC to start the calibration. ++")

        while True:
            frame, ret = self._videostream.read_ret()
            if not ret:
                print("failed to grab frame")
                break

            cv2.imshow("Take photo of chessboard", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Closing photo cam. Start calibration")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "Calibration_stuff/calibration_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        # self._videostream.pause_stream()

        cv2.destroyAllWindows()

        return self.check_board(cols, rows, img_name, debug)

    def check_board(self,
                    cols: int = 0,
                    rows: int = 0,
                    image_name: str = "",
                    debug: bool = False
                    ):
        """Calibration via chessboards with opencv

        :param cols: columns of the chessborad to be detected
        :param rows: rows of the chessboard to be detected
        :param image: name of the file the photo of he chessboard is saved as
        :param debug: debug mode

        :return: if detection was successful
        """

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

        # Object points (preparation)
        object_points = np.zeros((cols * rows, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

        # Storage
        real_points = []
        img_points = []

        image = cv2.imread(image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if debug:
            cv2.imshow('img', image)

        # Find chessboard corners
        results = cv2.findChessboardCorners(gray, (rows, cols), None)
        ret = results[0]
        corners = results[1]

        if debug:
            print("results: ")
            print(results)

        # If found add object points
        if ret is True:
            real_points.append(object_points)

            corners2 = cv2.cornerHarris(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            # Draw + display
            cv2.drawChessboardCorners(image, (rows, cols), corners2, ret)
            cv2.imshow('img', image)
            cv2.waitKey(500)

            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)

            frame1 = self._videostream.read()
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._width, self._height))

            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # undistort
            dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv2.imwrite('calibresult.png', dst)

        return ret

    def detect(self,
               detect: list = [],
               no_detect: list = [],
               functionality: Functionality = Functionality.Detection,
               autosave: bool = False,
               video_title: str = "",
               debug: bool = False
               ):
        """
        Object detection via a live camera feed and distance calculations of the objects.
        """

        if (autosave):
            if (debug):
                print("==== AUTOSAVE ON ====")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter(video_title + ".avi", fourcc, 20.0, (640, 480))

        color_variation = 0
        distance_calc_results = []

        # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

            objects_count = 0

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = self._videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
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
            scores = self._interpreter.get_tensor(self._output_details[2]['index'])[
                0]  # Confidence of detected objects

            coords = []
            center_coordinates = []

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
                    if object_name != "bottle":
                        continue

                    objects_count += 1

                    # Get bounding box coordinates
                    # Interpreter can return coordinates that are outside of image dimensions,
                    # need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * self._imH)))
                    xmin = int(max(1, (boxes[i][1] * self._imW)))
                    ymax = int(min(self._imH, (boxes[i][2] * self._imH)))
                    xmax = int(min(self._imW, (boxes[i][3] * self._imW)))

                    center = [round((xmin + xmax)/2), round((ymin + ymax)/2)]

                    # Save coordinates of detected person
                    coords.append([[xmin, ymin], [xmax, ymax]])
                    center_coordinates.append(center)

                    # Boxes color variation
                    if (i + 1) * 40 > 255:
                        color_variation += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (10, (40 + (40 * i)) % 255, (color_variation * 40) % 255), 2)

                    if functionality == Functionality.Distance:
                        # Distance calculation
                        if len(coords) >= 1:
                            distance_calc_results = self._distance_calculation(object_coordinates=coords,
                                                                               center_coordinates=center_coordinates,
                                                                               distance_threshold=self.distance_threshold,
                                                                               width_of_obj=self.objects_width_cm,
                                                                               debug=debug)
                            # Returns index of first object, index of second object, distance in pixel, distance in cm

                            if (debug):
                                print(distance_calc_results)

                            if distance_calc_results is None:
                                raise ValueError("Distance calculation results is None.")

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
            elif functionality == Functionality.Distance:
                list_of_drawn_lines = []
                for listelem in distance_calc_results:
                    index1 = listelem[0]
                    index2 = listelem[1]

                    # Smallest index is first
                    if index2 < index2:
                        temp = index1
                        index1 = index2
                        index2 = temp

                    # Check for duplicates
                    id = str(index1)+'-'+str(index2)
                    if debug:
                        print('ID: '+id)
                    if id in list_of_drawn_lines:
                        continue
                    list_of_drawn_lines.append(id)

                    # Call for drawing the line
                    try:
                        point1 = tuple(listelem[4][index1])
                        point2 = tuple(listelem[4][index2])
                    except IndexError:
                        print("!! -- "+center_coordinates+" -- !!")
                    self._draw(frame, point1, point2, listelem[3], self.distance_threshold)

            # Draw framerate in corner of frame
            cv2.putText(frame, 'DEMO // FPS: {0:.2f}'.format(self._frame_rate_calc), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2, cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / self._freq
            self._frame_rate_calc = 1 / time1

            if (autosave):
                video_out.write(frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('w'):
                self._videostream.pause_stream()
            elif cv2.waitKey(1) == ord('s'):
                self._videostream.continue_stream()

        if (autosave):
            video_out.release()

        self._videostream.stop_stream()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        Destructor.
        """

        # Clean up
        self._videostream.stop_stream()
        cv2.destroyAllWindows()