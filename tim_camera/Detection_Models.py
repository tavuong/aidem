"""Specified Models for object detection with tensorflow and distance calculation."""

from Detection_Basis import Detection

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

# Import packages
import os
import cv2.cv2 as cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import glob


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Based on - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the camera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        self.width = self.stream.get(3)

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

        self.stopped = False
        self.start()

    def stop_stream(self):
        # Indicate that stream should be released
        self.release_stream = True


class LiveDetection(Detection):
    """
    Class for live object detection
    """

    def __init__(self,
                 model_name:str = 'Sample_Model',
                 graph_name:str = 'detect.tflite',
                 labelmap_name:str = 'labelmap.txt',
                 min_conf_threshold:int = 0.5,
                 use_tpu:str = '',
                 distance_threshold:int = 150,
                 resolution:str = '1280x720',
                 debug:bool = False
                 ):
        """
        Live object detection and distance calculation.

        @param model_name: Name of the directory for the detection model
        @param graph_name: Name of the used detection model file
        @param labelmap_name: Name of the used labelmap file
        @param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        @param use_tpu: specifier if a TPU is to be used
        @param distance_threshold: minimum distance value between objects
        @param resolution: desired video resolution
        """

        super(LiveDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold, use_tpu, distance_threshold)

        resW, resH = resolution.split('x')
        self._imW, self._imH = int(resW), int(resH)


        # Initialize frame rate calculation
        self._frame_rate_calc = 1
        self._freq = cv2.getTickFrequency()

        # Initialize video stream
        self._videostream = VideoStream(resolution=(self._imW, self._imH), framerate=30).start()
        time.sleep(1)


    def calibrate(self,
                  obj_width_cm: int = 0,
                  obj_dist_cm: int = 0,
                  obj_name: str = "",
                  sample_count:int = 10,
                  debug:bool = False
                  ):
        """
        Calculation for the focal width of used camera. Note that a change in the focal width after calibrating
        will result in faulty distance calculations.

        @param obj_width_cm: the width of the object to calibrate with
        @param obj_dist_cm: the distance of the object to calibrate with
        @param obj_name: the name of the object to calibrate with
        @param sample_count: number of samples to take for focal width calculation (mean will be used)
        @return: True as signal for GUI

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
                if ((scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0)):

                    # Check for the right object (ensure correct measurement when several objects are detected)
                    if (self._labels[int(classes[i])] != obj_name):
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
        if(debug):
            print("Focal value:",self.focal_value)

        return True


    def calibrate_board(self,
                        cols: int = 0,
                        rows: int = 0,
                        debug:bool = False
                        ):
        """Calibration via chessboard with opencv

        @param cols: columns of the chessborad to be detected
        @param rows: rows of the chessboard to be detected
        @param debug: debug mode

        @return: if calibration was successful
        """

        img_counter = 0
        img_name = ""

        while True:
            frame, ret = self._videostream.read_ret()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Closing photo cam. Start calibration")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "Calibration_stuff/calibration_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        #self._videostream.pause_stream()

        cv2.destroyAllWindows()

        return self.check_board(cols, rows, img_name, debug)



    def check_board(self,
                        cols: int = 0,
                        rows: int = 0,
                        img_name: str = "",
                        debug:bool = False
                        ):
        """Calibration via chessboards with opencv

        @param cols: columns of the chessborad to be detected
        @param rows: rows of the chessboard to be detected
        @param img_name: name of the file the photo of he chessboard is saved as
        @param debug: debug mode

        @return: if detection was successful
        """

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Object points (preparation)
        object_points = np.zeros((cols*rows,3), np.float32)
        object_points[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

        # Storage
        real_points = []
        img_points = []

        # Read image
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)

        # If found add object points
        if ret is True:
            real_points.append(object_points)

            corners2 = cv2.cornerHarris(gray, corners, (11,11), (-1, -1), criteria)
            img_points.append(corners)

            # Draw + display
            cv2.drawChessboardCorners(img, (cols,rows), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)

            frame1 = self._videostream.read()
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._width, self._height))

            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite('calibresult.png', dst)

        return ret



    def detect(self,
               detect: list = [],
               no_detect: list = [],
               autosave: bool = False,
               video_title: str = "",
               debug:bool = False
               ):
        """
        Object detection via a live camera feed and distance calculations of the objects.
        """

        if(autosave):
            if(debug):
                print("==== AUTOSAVE ON ====")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter(video_title+".avi", fourcc, 20.0, (640,480))

        color_variation = 0

        # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

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
            # num = self._interpreter.get_tensor(self._output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            coords = []

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0)):

                    object_name = self._labels[int(classes[i])]

                    if(len(detect) > 0):
                        if object_name not in detect:
                            continue
                    if(len(no_detect) > 0):
                        if object_name in no_detect:
                            continue

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * self._imH)))
                    xmin = int(max(1, (boxes[i][1] * self._imW)))
                    ymax = int(min(self._imH, (boxes[i][2] * self._imH)))
                    xmax = int(min(self._imW, (boxes[i][3] * self._imW)))

                    if (i + 1) * 40 > 255:
                        color_variation += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (10, (40 + (40 * i)) % 255, (color_variation * 40) % 255), 2)

                    # Save coordinates of detected person
                    coords.append([[xmin, ymin], [xmax, ymax]])

                    if len(coords) >= 1:

                        if object_name == "person":
                            result = self._distance_calculation(coords, debug=debug)
                        else:
                            result = self._distance_calculation(coords, debug=debug)

                        if(debug):
                            print(result)

                        if result is None:
                            raise Exception("Distance calculation results in None.")
                        elif result[0] == 3:
                            pass
                        elif result[0] == 0:
                            self._draw(frame, coords, i, result[1], result[2], result[3])
                        elif result[0] == 1:
                            self._draw(frame, coords, i, result[2], result[1], result[3])
                        elif result[0] == 2:
                            pass
                        else:
                            raise Exception("Invalid distance calculation result.")

                    else:
                        # ...
                        b = 3

                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame, 'FPS: {0:.2f}'.format(self._frame_rate_calc), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2, cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / self._freq
            self._frame_rate_calc = 1 / time1

            if(autosave):
                video_out.write(frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('w'):
                self._videostream.pause_stream()
            elif cv2.waitKey(1) == ord('s'):
                self._videostream.continue_stream()

        if(autosave):
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


class VideoDetection(Detection):
    """
        Class for video object detection
        """

    def __init__(self,
                 model_name:str = 'Sample_Model',
                 graph_name:str = 'detect.tflite',
                 labelmap_name:str = 'labelmap.txt',
                 min_conf_threshold:int = 0.5,
                 use_tpu:str = '',
                 distance_threshold:int = 150,
                 debug:bool = False
                 ):
        """
        Video object detection and distance calculation.

        @param model_name: Name of the directory for the detection model
        @param graph_name: Name of the used detection model file
        @param labelmap_name: Name of the used labelmap file
        @param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        @param use_tpu: specifier if a TPU is to be used
        @param distance_threshold: minimum distance value between objects
        """

        super(VideoDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold, use_tpu, distance_threshold)

        self._video = None
        self.stop = False


    def detect(self,
               video_name: str = "Sample_Video/testvideo1.mp4",
               focal_width: int = 1000,
               debug:bool = False
               ):
        """
        Object detection via a video feed and distance calculations of the objects.

        @param video_name: path to the video that should be used for detection
        @param focal_width: focal width of the used camera in the video
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
            classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]  # Class index of detected objects
            scores = self._interpreter.get_tensor(self._output_details[2]['index'])[0]  # Confidence of detected objects
            # num = interpreter.get_tensor(self.output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            coords = []

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0)):

                    object_name = self._labels[int(classes[i])]

                    if (object_name != "person" and object_name != "teddy bear" and object_name != "chair"):
                        continue

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

                    if (len(coords) > 1):

                        if object_name == "person":
                            result = self._distance_calculation(coords, debug=debug)
                        else:
                            result = self._distance_calculation(coords, debug=debug)

                        if(debug):
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

                    else:
                        # ...
                        b = 3

                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

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
