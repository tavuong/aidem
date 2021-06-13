"""Video object detection and distance calculation """

__version__ = "0.1.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz@stud.uni-frankfurt.de"
__credits__ = "Special thanks to The Anh Vuong who came up with the original idea." \
              "This code is also based off of the code from Evan Juras (see below)"

######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import math


class VideoDetection():
    """ """
    
    def __init__(self):
        """
        """
        
        self.MODEL_NAME = 'Sample_Model'
        self.GRAPH_NAME = 'detect.tflite'
        self.LABELMAP_NAME = 'labelmap.txt'
        self.min_conf_threshold = 0.5
        self.use_TPU = ''

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (self.GRAPH_NAME == 'detect.tflite'):
                self.GRAPH_NAME = 'edgetpu.tflite'   

        # Get path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.__labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.__labels[0] == '???':
            del(self.__labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(self.PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5
        
        self.avg_width_person = 45+8+4 # +8 due to borders not aligning to body
        self.avg_height_person = 172
        self.avg_proportion_person = self.avg_width_person / self.avg_height_person

        self.test_distance = 216
        
        # Variable for new calibrated value:
        self.focal_value = 2000


    def detect(self,
               video_name:str = "Sample_Video/testvideo1.mp4",
               focal_width:int = 1000
               ):
        """ """

        self.focal_value = focal_width
        
        # Path to video file
        VIDEO_PATH = os.path.join(self.CWD_PATH,video_name)
        
        color_variation = 0;
        
        # Open video file
        video = cv2.VideoCapture(VIDEO_PATH)
        imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while(video.isOpened()):

            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = video.read()
            if not ret:
              print('Reached the end of the video!')
              break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()

            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(self.output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
            
            coords = []
            proportion_x = []
            proportion_y = []
            camera_distance = []
            obj_type = []
            
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                    
                    if(self.__labels[int(classes[i])] != "person" and self.__labels[int(classes[i])] != "teddy bear" and self.__labels[int(classes[i])] != "chair"):
                        continue
                    else:
                        obj_type.append(str(self.__labels[int(classes[i])]))

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    if (i+1)*40 > 255:
                        color_variation += 1
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, (40+(40*i))%255, (color_variation*40)%255), 2)
                    
                    # Save coordinates of detected person
                    coords.append([[xmin, ymin],[xmax, ymax]])
                    
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

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        def __del__(self):

            # Clean up
            video.release()
            cv2.destroyAllWindows()


def main():
    """ """
    
    vid_det = VideoDetection()
    vid_det.detect()
    del vid_det


if __name__ == "__main__":
    main()
