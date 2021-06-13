"""
Program to do dct / idct with object detection.

Original script from Dr. The Anh Vuong
Modified by Tim Rosenkranz
Last updated: 2021-04-06
"""

import os
import numpy as np
import math
import matplotlib.pylab as plt

import cv2.cv2 as cv2
import importlib.util
from PIL import Image
import imageio
import lib.imgRW as irw
import lib.Process_2D as P2D
import lib.bloc_process as bp
import lib.live_process as lp
import config


# from lib.dct import dct_2d, dct_1d
# from lib.idct import idct_2d, idct_1d
# from scipy import fftpack


class Detection:
    """
    Class for detection models basis
    """

    def __init__(self,
                 model_path: str = './lib/Sample_Model',
                 graph_name: str = 'detect.tflite',
                 labelmap_name: str = 'labelmap.txt',
                 min_conf_threshold: int = 0.5,
                 ):
        """
        Basis model for video detection and object distance calculations.

        @param model_path: Name of the directory for the detection model
        @param graph_name: Name of the used detection model file
        @param labelmap_name: Name of the used labelmap file
        @param min_conf_threshold: minimum confidence value for detected objects to be acknowledged
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
        path_to_ckpt = os.path.join(self._cwd_path, model_path, graph_name)


        # Path to label map file
        path_to_labels = os.path.join(self._cwd_path, model_path, labelmap_name)

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

        # Read image height and width
        reading_img = config.imageToRead
        reading_img = reading_img.split("/")[-1]
        reading_img = reading_img.split(".")[0]
        pixel = ""
        for char in reading_img:
            if char.isdigit():
                pixel += char
        if len(pixel) == 0:
            pixel = 0
        pixel = int(pixel)

        self._imH = pixel
        self._imW = pixel

        self.detected_obj_matrix = []

    def detect(self, img_name: str, img_num: int = -1, block_type: str = "None"):
        """ Perform detection

        @:param img_name: Path of the image to detect
        @param img_num: number of the block image (0, 1 , 2, ...)
        @param block_type: type of the block image (e.g. original, spectrum, filter)

        @return wether an object has been detected in the given image (True or False)
        """

        detected = False
        detected_obj = [img_num, block_type]

        img = cv2.imread(img_name)

        frame = img.copy()
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
        # Bounding box coordinates of detected objects
        boxes = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        # Class index of detected objects
        classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]
        # Confidence of detected objects
        scores = self._interpreter.get_tensor(self._output_details[2]['index'])[0]
        # Total number of detected objects (inaccurate and not needed)
        # num = self._interpreter.get_tensor(self._output_details[3]['index'])[0]

        for i in range(len(scores)):
            # Check scores
            if (scores[i] > self._min_conf_threshold) and (scores[i] <= 1.0):
                detected = True

                # Get name of detected object
                object_name = self._labels[int(classes[i])]

                """
                ymin = int(max(1, (boxes[i][0] * self._imH)))
                xmin = int(max(1, (boxes[i][1] * self._imW)))
                ymax = int(min(self._imH, (boxes[i][2] * self._imH)))
                xmax = int(min(self._imW, (boxes[i][3] * self._imW)))
                """

                detected_obj.append(object_name)

        self.detected_obj_matrix.append(detected_obj)
        return detected

    def log_detection(self, path: str = config.dirDetection):
        """Logs the detection results in a .log file"""

        writing_lines = ["block nr."+"\t"+"block type"+"\t"+"detection results"+"\n"]

        for iteration in self.detected_obj_matrix:
            new_line = ""
            for elem in iteration:
                new_line += str(elem) + "\t"
            writing_lines.append(new_line[:-1] + "\n")

        with open(path + "/detection.log", 'wt') as file:
            file.writelines(writing_lines)

        print("Detected objects:")
        print(self.detected_obj_matrix)


def imgBLOCK_tim(img, img2, numberBlock=0, param=0, nbit=0, blackout: bool = False, black_blocks: int = 0):
    """Modified imgBlock script with hooked object detection

    @param img: image matrix
    @param img2: image matrix 2
    @param numberBlock: Number of blocks the image will be divided into
    @param param: parameter setting for animation (deactivated in this method)
    @param nbit: number of bits of the image
    @param blackout: Determine whether blocks with no detectable object in them should be blacked out
    @param black_blocks: Determine at what iteration the blocks should be blacked (orig - 0, spect - 1, filter - 2, all - 5)
    """

    detector = Detection()
    # Path were outputs will be saved to
    detection_path = config.dirDetection + "/"

    #	param = 0 : komplet image stored
    #   param = 1 : block-spechts- reconst stored
    nb = numberBlock
    para = param
    nmap = nbit
    height = img.shape[0]
    width = img.shape[1]
    nzahl = int(height / nb)
    #	print("pic height:" + str(height))
    #	print("pic width:" + str(width))
    #	print("block:" + str(nb))
    #	print("Param:" + str(para))
    #   ------------ Preset Mini-Block ---------------------
    #   preset ordner
    #	os.mkdir("./blocks/")
    imageBlock = np.zeros_like(img2).astype(float)
    #	imageSpect = np.zeros_like(img2).astype(float)
    #	imageFilter = np.zeros_like(img2).astype(float)
    #	imageRecon = np.zeros_like(img2).astype(float)
    #   ------------ Preset full-Image ---------------
    imgReconful = np.zeros_like(img).astype(float)
    imgSpectful = np.zeros_like(img).astype(float)
    imgFilterful = np.zeros_like(img).astype(float)

    #   Preset full-image
    #	filename_save = config.dirImgout + '/test-'
    #	ps = P2D.Block2D(img,filename_save)
    #	imgReconful2 = ps.copy_block()
    #	precont1 = P2D.Block2D(imgReconful2,filename_save)
    #	precont1.save_image(2000)

    pspect = P2D.Block2D(imgSpectful)

    pfilter = P2D.Block2D(imgFilterful)

    precont = P2D.Block2D(imgReconful)

    ncof = 0
    ncof = int(nb * nb)
    nf = 0
    nbK = 0
    nbK1 = nb
    nbI = 0
    nbI1 = nb
    #   ------------ Pipe Processing ---------------------
    #
    # Blockweise processing implemented
    # ORI >>> DCT >>> FILTER >>> IDCT >>> RECONST

    for ri in range(nzahl):
        for r in range(nzahl):

            # -----------------ORI -------------
            for i in range(nbI, nbI1):
                for k in range(nbK, nbK1):
                    imageBlock[i - nbI][k - nbK] = img[i][k]

            # print ("Block calculated "+ str(nf) + "\r")
            # Save in t_blocks folder
            ps = P2D.Block2D(imageBlock)
            ps.print("Block calculated " + str(nf))
            filename_save = detection_path + '/block'
            ps.save_image(nf, filename_save)

            # Detection
            result = detector.detect(filename_save + str(nf) + ".png", nf, "[orig]")
            if result:
                print("An object detected in orig block number " + str(nf))
            elif blackout and not result and (black_blocks == 0 or black_blocks == 5):
                ps.blackout_block()

            # -----------------Spectrum  -------------
            imageSpect = ps.dct_2D()
            ps = P2D.Block2D(imageSpect)
            filename_save = detection_path + '/spect'
            ps.save_image(nf, filename_save)

            # Detection
            result = detector.detect(filename_save + str(nf) + ".png", nf, "[spect]")
            if result:
                print("An object detected in spect block number " + str(nf))
            elif blackout and not result and (black_blocks == 1 or black_blocks == 5):
                ps.blackout_block()

            # -----------------2D-Filter  -------------
            imageFilter = ps.lowpass_2D(nmap)
            ps = P2D.Block2D(imageFilter)
            filename_save = detection_path + '/filter'
            ps.save_image(nf, filename_save)

            # Detection
            result = detector.detect(filename_save + str(nf) + ".png", nf, "[filter]")
            if result:
                print("An object detected in filter block number " + str(nf))
            elif blackout and not result and (black_blocks == 2 or black_blocks == 5):
                ps.blackout_block()

            # -----------------IDCT   -------------
            imageRecont = ps.idct_2D()
            filename_save = detection_path + '/recon'
            ps = P2D.Block2D(imageRecont)
            ps.save_image(nf, filename_save)

            # Detection
            # result = detector.detect(filename_save + str(nf) + ".png", nf)
            # if result:
            #    print("An object detected in recon block number " + str(nf))

            # ---nf = Block index
            nf = nf + 1
            # --------------------Block Fill to Image -----------------------------
            #			for i in range (nbI, nbI1):
            #				for k in range (nbK, nbK1):
            #					imgReconful[i][k]=imageRecont[i-nbI][k-nbK]
            #					imgSpectful[i][k]=imageSpect[i-nbI][k-nbK]
            #					imgFilterful[i][k]=imageFilter[i-nbI][k-nbK]

            pspect.refill(imageSpect, nbI, nbI1, nbK, nbK1)
            pfilter.refill(imageFilter, nbI, nbI1, nbK, nbK1)
            precont.refill(imageRecont, nbI, nbI1, nbK, nbK1)
            #			precont1.refill (imageFilter,nbI,nbI1,nbK,nbK1)
            nbK = nbK + nb
            nbK1 = nbK1 + nb
        nbI = nbI + nb
        nbI1 = nbI1 + nb
        nbK = 0
        nbK1 = nb
    #	cv2.imwrite(config.dirImgout + '/spect-' + str(nf)+ '.jpg',imgSpectful)
    #	cv2.imwrite(config.dirImgout + '/filter-' + str(nf)+ '.jpg',imgFilterful)
    #	cv2.imwrite(config.dirImgout + '/reconst-' + str(nf)+ '.jpg',imgReconful)

    #	precont1.save_image(100)

    # ---- Save full immages

    filename_save = detection_path + '/spect-'
    ps = P2D.Block2D(imgSpectful)
    ps.save_image_jpg(nf, filename_save)

    filename_save = detection_path + '/filter-'
    ps = P2D.Block2D(imgFilterful)
    ps.save_image_jpg(nf, filename_save)

    filename_save = detection_path + '/reconst-'
    ps = P2D.Block2D(imgReconful)
    ps.save_image_jpg(nf, filename_save)

    detector.log_detection()

    #ishow = irw.imgSHOW2(nf)
    return imageBlock

