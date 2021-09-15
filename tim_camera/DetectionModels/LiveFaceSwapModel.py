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
import time
import dlib

from aidem.tim_camera.DetectionModels.BasisModel import VideoStream


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

        :param resolution: desired video resolution (<width>x<height>)
        :param debug: Option to show debug information
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


    def face_swap(self, swap_img="barack_obama.jpeg"):
        """
        Live face swapping

        :param swap_img: The source of the image to swap face with
        """

        landmarks_points = None
        landmarks_points2 = None

        img = cv2.imread("./swap_faces/"+swap_img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)

        # Load face detector and facial landmarks shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        indexes_triangles = []

        # Face 1: Face to swap into
        # Detect faces in the image
        faces = detector(img_gray)
        for face in faces:
            # Detect the landmarks
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for ii in range(0, 68):
                # Collect the landmark points coordinates
                coordinates = [landmarks.part(ii).x, landmarks.part(ii).y]
                landmarks_points.append(coordinates)

            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, convexhull, 255)

            # Delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            indexes_triangles = []
            for t in triangles:
                # Triangle indices
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                # Extract landmark indices
                landmark_pt1 = np.where((points == pt1).all(axis=1))[0][0]
                landmark_pt2 = np.where((points == pt2).all(axis=1))[0][0]
                landmark_pt3 = np.where((points == pt3).all(axis=1))[0][0]

                if landmark_pt1 is not None and landmark_pt2 is not None and landmark_pt3 is not None:
                    triangle = [landmark_pt1, landmark_pt2, landmark_pt3]
                    indexes_triangles.append(triangle)


        while True:
            img2 = cv2.imread("./swap_faces/"+"barack_obama.jpeg")#self._videostream.read()
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            new_face = np.zeros_like(img2)

            # Face 2
            faces2 = detector(img2_gray)
            for face in faces2:
                landmarks = predictor(img2_gray, face)
                landmarks_points2 = []
                for ii in range(0, 68):
                    # Collect the landmark points coordinates
                    coordinates = [landmarks.part(ii).x, landmarks.part(ii).y]
                    landmarks_points2.append(coordinates)

                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)

            if landmarks_points is None or landmarks_points2 is None:
                # No face found in both or either picture
                continue
            # Triangulation of both faces
            for triangle_index in indexes_triangles:
                # Extract the landmark points for the triangle for each face
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                # Triangulation of the first face
                # Create rectangle around the coordinates for the triangle
                rect1 = cv2.boundingRect(triangle1)
                # Extract coordinates (x, y) and height and width
                (x, y, w, h) = rect1
                # Cut the rectangle into a triangle
                cropped_triangle = img[y: y + h, x: x + w]
                # Create the triangle mask
                cropped_triangle_mask_1 = np.zeros((h, w), np.uint8)

                # Calculate the points to fill in the mask
                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                # Fill
                cv2.fillConvexPoly(cropped_triangle_mask_1, points, 255)

                # Triangulation of second face
                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2

                cropped_triangle_mask_2 = np.zeros((h, w), np.uint8)

                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_triangle_mask_2, points2, 255)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                transformation_matrix = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, transformation_matrix, (w, h))
                # Smooth edges
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_triangle_mask_2)

                # Reconstructing destination face
                new_face_rect_area = new_face[y: y + h, x: x + w]
                new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                # Add
                new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
                new_face[y: y + h, x: x + w] = new_face_rect_area


            # Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(img2_gray)
            # Mask the face in the livestream image
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            # Invert the mask so the face is cut out
            img2_face_mask = cv2.bitwise_not(img2_head_mask)

            # Direct swap
            img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, new_face)

            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

            # Seamless swap
            seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

            # Demo label
            cv2.putText(img2, 'DEMO', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(seamlessclone, 'DEMO', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(result, 'DEMO', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("img2", img2)
            cv2.imshow("clone", seamlessclone)
            cv2.imshow("result", result)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self._videostream.stop_stream()
        cv2.destroyAllWindows()