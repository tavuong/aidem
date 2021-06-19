"""Specified Models for object detection with tensorflow and distance calculation."""

__version__ = "1.0.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz:stud.uni-frankfurt.de"
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
import dlib
import importlib.util
import glob
import typing

from Detection_Basis import Detection


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
                 model_name: str = 'Sample_Model',
                 graph_name: str = 'detect.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.5,
                 use_tpu: str = '',
                 distance_threshold: int = 150,
                 resolution: str = '1280x720',
                 debug: bool = False
                 ):
        """
        Live object detection and distance calculation.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param use_tpu: specifier if a TPU is to be used
        :param distance_threshold: minimum distance value between objects
        :param resolution: desired video resolution
        """

        super(LiveDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold, use_tpu,
                                            distance_threshold)

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

        img_counter = 0
        img_name = ""

        print("++ Press SPACE to take a photo of the chess board. Press ESC to start the calibration. ++")

        while True:
            frame, ret = self._videostream.read_ret()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)

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

        return self.check_board(cols, rows, frame, debug)

    def check_board(self,
                    cols: int = 0,
                    rows: int = 0,
                    image: any = None,
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

        # Read image
        #img = cv2.imread(img_name)
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
            cv2.drawChessboardCorners(image, (cols, rows), corners2, ret)
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

                    if (len(detect) > 0):
                        if object_name not in detect:
                            continue
                    if (len(no_detect) > 0):
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

                        if (debug):
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


class LiveFaceSwap():
    """
    Class for live detection connected to a convolutional net
    """

    def __init__(self,
                 resolution: str = '1280x720',
                 debug: bool = False
                 ):
        """
        Live object detection and distance calculation.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param use_tpu: specifier if a TPU is to be used
        :param distance_threshold: minimum distance value between objects
        :param resolution: desired video resolution (<width>x<height>)
        """

        resW, resH = resolution.split('x')
        self._imW, self._imH = int(resW), int(resH)

        # Initialize frame rate calculation
        self._frame_rate_calc = 1
        self._freq = cv2.getTickFrequency()

        # Initialize video stream
        self._videostream = VideoStream(resolution=(self._imW, self._imH), framerate=30).start()
        time.sleep(1)

    def detect(self,
               cascPath: str = "haarcascade_frontalface_default.xml",
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
        faceCascade = cv2.CascadeClassifier(cascPath)

        while True:
            # read and convert image
            image = self._videostream.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            #print("Found {0} faces!".format(len(faces)))

            # show face detections
            i = 0
            for (x, y, w, h) in faces:
                if (i + 1) * 40 > 255:
                    color_variation += 1
                cv2.rectangle(image, (x, y), (x+w, y+h), (10, (40 + (40 * i)) % 255, (color_variation * 40) % 255), 2)
                i += 1

            cv2.imshow("Face detector", image)

            if (autosave):
                video_out.write(image)

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


    def face_swap(self, swap_img="barack_obama.jpeg"):#frame=None, newface=None):
        """
        Live face swapping

        :param swap_img: The source of the image to swap face with
        """

        landmarks_points = None
        landmarks_points2 = None

        def extract_index_nparray(nparray):
            index = None
            for num in nparray[0]:
                index = num
                break
            return index

        img = cv2.imread("./swap_faces/"+swap_img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
    
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
        indexes_triangles = []
    
        # Face 1
        faces = detector(img_gray)
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
    
                # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
            cv2.fillConvexPoly(mask, convexhull, 255)
    
            face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    
            # Delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)
    
            indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
    
                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = extract_index_nparray(index_pt1)
    
                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = extract_index_nparray(index_pt2)
    
                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = extract_index_nparray(index_pt3)
    
                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    indexes_triangles.append(triangle)
    
    
        while True:
            img2 = self._videostream.read()
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2_new_face = np.zeros_like(img2)
    
            # Face 2
            faces2 = detector(img2_gray)
            for face in faces2:
                landmarks = predictor(img2_gray, face)
                landmarks_points2 = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points2.append((x, y))
    
                # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)
    
            lines_space_mask = np.zeros_like(img_gray)
            lines_space_new_face = np.zeros_like(img2)

            if landmarks_points is None or landmarks_points2 is None:
                continue
            # Triangulation of both faces
            for triangle_index in indexes_triangles:
                # Triangulation of the first face
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    
                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    
                # Triangulation of second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2
    
                cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    
                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
    
    
                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    
    
                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    
                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    
            # Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(img2_gray)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)
    
    
            img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, img2_new_face)
    
            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    
            seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
    
            cv2.imshow("img2", img2)
            cv2.imshow("clone", seamlessclone)
            cv2.imshow("result", result)
    
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    
        self._videostream.stop_stream()
        cv2.destroyAllWindows()


class VideoDetection(Detection):
    """
        Class for video object detection
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
        Video object detection and distance calculation.

        :param model_name: Name of the directory for the detection model
        :param graph_name: Name of the used detection model file
        :param labelmap_name: Name of the used labelmap file
        :param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
        :param use_tpu: specifier if a TPU is to be used
        :param distance_threshold: minimum distance value between objects
        """

        super(VideoDetection, self).__init__(model_name, graph_name, labelmap_name, min_conf_threshold, use_tpu,
                                             distance_threshold)

        self._video = None
        self.stop = False

    def detect(self,
               video_name: str = "Sample_Video/testvideo1.mp4",
               focal_width: int = 1000,
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
