""" Webcam Detection with Tensorflow calssifier and object distance calculation """

__version__ = "0.1.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz@stud.uni-frankfurt.de"
__credits__ = "Special thanks to The Anh Vuong who came up with the original idea." \
              "This code is also based off of the code from Evan Juras (see below)"

# This script is based off of a script by Evan Juras (see below).
# I rewrote this script to be object oriented and added the tkinter-ui (removed command
# line functionalities) as well as several functionalities to calculate the distance
# between two detected object

######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I [Evan Juras] added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import math

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        
        self.width = self.stream.get(3)
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
    def continue_video(self):
        # Indicate that camera should resume
        
        self.stopped = False
        self.start()


class LiveDetection:
    """

    """
    
    def __init__(self):
        """

        """

        MODEL_NAME = 'Sample_Model'
        GRAPH_NAME = 'detect.tflite'
        LABELMAP_NAME = 'labelmap.txt'
        self.__min_conf_threshold = 0.5
        resW, resH = '1280x720'.split('x')
        self.__imW, self.__imH = int(resW), int(resH)
        use_TPU = ''

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            self.__labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.__labels[0] == '???':
            del(self.__labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            self._interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            self._interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self._interpreter.allocate_tensors()

        
        # Get model details
        self.__input_details = self._interpreter.get_input_details()
        self.__output_details = self._interpreter.get_output_details()
        self.__height = self.__input_details[0]['shape'][1]
        self.__width = self.__input_details[0]['shape'][2]

        self.__floating_model = (self.__input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Initialize frame rate calculation
        self.__frame_rate_calc = 1
        self.__freq = cv2.getTickFrequency()

        # Initialize video stream
        self._videostream = VideoStream(resolution=(self.__imW,self.__imH),framerate=30).start()
        time.sleep(1)

        # -----------------------------------------------------------------
        # Average parameters
        self.avg_width_person = 45+8+4 # +8 due to borders not aligning to body
        self.avg_height_person = 172
        self.avg_proportion_person = self.avg_width_person / self.avg_height_person

        self.test_distance = 216
        
        # Old value:
        self.fokal_empir = 1500
        
        # Variable for new calibrated value:
        self.focal_value = 0
    
    def calibrate(self,
                  obj_width_cm:int=0,
                  obj_dist_cm:int=0,
                  obj_name:str=""
                  ):
        """

        """
        
        color_variation = 0
        foc_meas = 0
        
        for i in range(10):
            # Grab frame from video stream
            frame1 = self._videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.__width, self.__height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.__floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            self._interpreter.set_tensor(self.__input_details[0]['index'],input_data)
            self._interpreter.invoke()

            # Retrieve detection results
            boxes = self._interpreter.get_tensor(self.__output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self._interpreter.get_tensor(self.__output_details[1]['index'])[0] # Class index of detected objects
            scores = self._interpreter.get_tensor(self.__output_details[2]['index'])[0] # Confidence of detected objects
                
            obj_type = []

            for i in range(len(scores)):
                if ((scores[i] > self.__min_conf_threshold) and (scores[i] <= 1.0)):
                    
                    # Check for the right object (ensure correct measurement when several objects are detected)
                    if(self.__labels[int(classes[i])] != obj_name):
                        continue
                    else:
                        obj_type.append(str(self.__labels[int(classes[i])]))
                        
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * self.__imH)))
                    xmin = int(max(1,(boxes[i][1] * self.__imW)))
                    ymax = int(min(self.__imH,(boxes[i][2] * self.__imH)))
                    xmax = int(min(self.__imW,(boxes[i][3] * self.__imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, (40+(40*i))%255, (color_variation*40)%255), 2)
                    
                    # Calculate object width in pixel
                    obj_width_pixels = xmax - xmin
                    foc_meas += (obj_width_pixels * obj_dist_cm) / obj_width_cm
                        
                    # Draw label
                    object_name = self.__labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.__frame_rate_calc),(15,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
        
        self.focal_value = foc_meas / 10
        
        print("Calculated focal value:",self.focal_value)
        print("Calibration done")
            
    
    def detect(self):
        """

        """
        
        color_variation = 0;
        
        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = self._videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.__width, self.__height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.__floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            self._interpreter.set_tensor(self.__input_details[0]['index'],input_data)
            self._interpreter.invoke()

            # Retrieve detection results
            boxes = self._interpreter.get_tensor(self.__output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self._interpreter.get_tensor(self.__output_details[1]['index'])[0] # Class index of detected objects
            scores = self._interpreter.get_tensor(self.__output_details[2]['index'])[0] # Confidence of detected objects
            #num = self._interpreter.get_tensor(self.__output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
            
            # --------------------------------------------------------------------------------------------------------
            coords = []
            proportion_x = []
            proportion_y = []
            camera_distance = []
            obj_type = []
            
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.__min_conf_threshold) and (scores[i] <= 1.0)):
                    
                    if(self.__labels[int(classes[i])] != "person" and self.__labels[int(classes[i])] != "teddy bear" and self.__labels[int(classes[i])] != "chair"):
                        continue
                    else:
                        obj_type.append(str(self.__labels[int(classes[i])]))

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * self.__imH)))
                    xmin = int(max(1,(boxes[i][1] * self.__imW)))
                    ymax = int(min(self.__imH,(boxes[i][2] * self.__imH)))
                    xmax = int(min(self.__imW,(boxes[i][3] * self.__imW)))
                    
                    if (i+1)*40 > 255:
                        color_variation += 1
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, (40+(40*i))%255, (color_variation*40)%255), 2)
                    
                    # Save coordinates of detected person
                    coords.append([[xmin, ymin],[xmax, ymax]])
                    
                    # For testing (screen width of camera)
                    vid_width = int(self._videostream.width)
                    
                    if(len(coords) > 1):
                        # preparation
                        for a in range(len(coords)):
                            proportion_x.append(0)
                            proportion_y.append(0)
                        for i in range(len(coords)):
                            # Measure height and width of detected person (in pixel)
                            proportion_x[i] = coords[i][1][0] - coords[i][0][0] # Width
                            #proportion_y[i] = coords[i][1][1] - coords[i][0][1] # Height
                            #proportion_x[i] = xmax - xmin
                            
                            # P = proportion_x[i]
                            
                            # F = Fokalwert, W = Objektbreite (cm), P = Objektbreite (Pixel), D = Distanz (cm)
                            # F = (P * D) / W -> D = (F * W) / P
                            
                            # F = (P * test_distance) / (45+8)
                            # print(F)
                            
                            # Calculate object distance to camera
                            
                            camera_distance.append((self.focal_value * self.avg_width_person) / proportion_x[i])
                            print("Distance obj "+str(i)+" ("+str(obj_type)+") - camera: "+str(camera_distance[i]), flush=True)
                            
                            if(i>0):
                                # Calculate min dist (only horizontal)
                                if(obj_type[i] == "person"):
                                    min_dist_x = proportion_x[i]/self.avg_width_person * 150
                                elif(obj_type[i] == "chair"):
                                    min_dist_x = proportion_x[i]/80 * 150
                                else:
                                    min_dist_x = 500
                                #min_dist_x = 300
                                for j in range(i):
                                    min_dist_obj_x_1 = abs(coords[i][1][0] - coords[j][0][0])
                                    min_dist_obj_x_2 = abs(coords[j][1][0] - coords[i][0][0])
                                    
                                    dist_obj_z = abs(camera_distance[i] - camera_distance[j])
                                    
                                    # Test with distance to borders
                                    #min_dist_obj_x_1 = abs(coords[i][1][0] - vid_width) # To the right
                                    #min_dist_obj_x_2 = abs(coords[i][0][0] - 0) # To the left
                                    
                                    print("X-Distanz objekt i -> j: "+str(min_dist_obj_x_1)+" - X-Distanz obj j -> i: "+str(min_dist_obj_x_2)+" - minimale Distanz: "+str(min_dist_x), flush=True)
                                    print("Z-Distanz objekt i - j: "+str(dist_obj_z), flush=True)
                                    
                                    # Check for smaller distance
                                    if(min_dist_obj_x_1 < min_dist_obj_x_2):
                                        objects_distance = math.sqrt(min_dist_obj_x_1**2 + dist_obj_z**2)
                                        if(objects_distance < min_dist_x):
                                            print("AAAA "+str(objects_distance)+" j = "+obj_type[j], flush=True)
                                            cv2.line(frame, (coords[i][1][0], coords[i][1][1]), (coords[j][0][0],coords[j][1][1]), (255,10,0), 2)
                                            #cv2.line(frame, (coords[i][1][0], coords[i][1][1]+30), (vid_width,coords[i][1][1]+30), (255,10,0), 2)
                                            
                                            dist_label = '%s / %d' % (round(objects_distance, 2), round(min_dist_x, 2))
                                            dist_labelSize, dist_baseLine = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                            dist_label_ymin = max(coords[i][1][1], dist_labelSize[1] + 10)
                                            cv2.rectangle(frame, (coords[i][1][0], dist_label_ymin-dist_labelSize[1]-10), (coords[i][1][0]+dist_labelSize[0], dist_label_ymin+dist_baseLine-10), (255, 255, 255), cv2.FILLED)
                                            cv2.putText(frame, dist_label, (coords[i][1][0], dist_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                    
                                    elif(min_dist_obj_x_1 > min_dist_obj_x_2):
                                        objects_distance = math.sqrt(min_dist_obj_x_2**2 + dist_obj_z**2)
                                        if(objects_distance < min_dist_x):
                                            print("BBB "+str(objects_distance)+" j = "+obj_type[j], flush=True)
                                            cv2.line(frame, (coords[j][1][0], coords[j][1][1]), (coords[i][0][0],coords[i][1][1]), (255,10,0), 2)
                                            #cv2.line(frame, (coords[i][0][0], coords[i][0][1]), (0,coords[i][0][1]), (255,10,0), 2)
                                            
                                            dist_label = '%s / %d' % (round(objects_distance, 2), round(min_dist_x, 2))
                                            dist_labelSize, dist_baseLine = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                            dist_label_ymin = max(coords[j][1][1], dist_labelSize[1] + 10)
                                            cv2.rectangle(frame, (coords[j][1][0], dist_label_ymin-dist_labelSize[1]-10), (coords[j][1][0]+dist_labelSize[0], dist_label_ymin+dist_baseLine-10), (255, 255, 255), cv2.FILLED)
                                            cv2.putText(frame, dist_label, (coords[j][1][0], dist_label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                            
                                    else:
                                        # ...
                                        b = 1
                            else:
                                # ...
                                b = 2
                    else:
                       # ...
                       b = 3
                    
                    # Draw label
                    object_name = self.__labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(self.__frame_rate_calc),(15,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.__freq
            self.__frame_rate_calc= 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            

    def __del__(self):
        """

        """
        
        # Clean up
        self._videostream.stop()
        cv2.destroyAllWindows()
        

def main():
    
    det_ob = LiveDetection()
    det_ob.detect()
    del det_ob
    

if __name__ == "__main__":
    main()

    