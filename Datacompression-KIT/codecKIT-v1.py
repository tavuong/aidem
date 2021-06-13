# CodecKIT.py
# Frame Work für block-codierung 
# Status: Entwicklung
# Demo- für die Codierung eines Bilders aus JPEG mit OPencv
# -Der Vorgang wird visulaisieren
# -Die Spektrum wird berechnet 
# -Die Matrix des Spektrum wird gefiltert, blockgrosse nbit
# - Reconstruierte Bild wird berechent   
# Alle Verarbeitungsschritten: Blocks, voll Bild als Jpg- file gepeichert, zu Visualisieren
# Systolic Methode: block zu block weiter verschieben und bearbeitet.
# Object Orientierte Programming 
# Datum : 12.04.2021
# Author: Dr.-Ing. The Anh Vuong 
# -*- coding: utf-8 -*-

import cv2.cv2  # openCV
import config
import lib.dash_board as board
import lib.bloc_process2 as bp2
import lib.bloc_process as bp1
import lib.live_process as lp
import lib.imgRW as irw
import lib.tim_process as timproc
import lib_1D.audio_process as audioproc
import lib.dct as dct
import lib.idct as idct
from math import cos, pi, sqrt
import numpy as np
import matplotlib.pylab as plt
import sys
#import lib.basis_image2 as bsproc

# Mangement for different Process
# Class Codec
# Datum : 12.04.2021
# Authors: Tim Orius, Dr.-Ing. The Anh Vuong 

""" 
class Codec:

    def __init__(self, number_block, nbit, animate: int = 1):
        Modus selection and preparation for codec

         @param number_block: length (pixel) per block.
         @param nbit: determine how large the filter should be.
         @param animate: determine whether the output should be animated.
        

        self.animate = animate
        self.number_block = number_block
        self.nbit = nbit
        self.number_coefficients = number_block * number_block

    def confirm(self):
        Confirmation of the initialization parameters

         @return confirmation input (True / False)
        

        print("image_input " + config.imageToRead + " !" + "\n")
        print("Blocklength: " + str(self.number_block) + " !" + "\n")
        print("Number of DCT-coefficients: " + str(self.number_coefficients) + " !" + "\n")
        print("2D LowpassFilter- Blocklength: " + str(self.nbit) + " !" + "\n")
        rcontinue = input("is setting correct (y /n)? ")
        return True if (rcontinue != "n") else False

    def process(self, img):
         Generates blocks in the set size with transformation and reconstruction with dct / idct.

        @param img: Bitwise image vector.
        

        # -------------------------------------
        # Blocks generator
        # dct  -> Spect- blocks
        # odct -> reconstr -blocks
        # blocks --> Big image back
        # from lib.blockprocess import imgBLOCK1
        # -------------------------------------

        a = np.arange(0, self.number_coefficients)
        b = a.reshape(self.number_block, self.number_block)
        img2 = np.zeros_like(b).astype(float)
        imgBLOCK = bp2.imgBLOCK2(img, img2, self.number_block, self.animate, self.nbit)
        print('*************** output ./img_out/movie*.gif ***********')
        if self.animate == 1:
            print('wait ...')
            animator_call = irw.animator(config.dirBlocks, "ori")
            animator_call = irw.animator(config.dirSpect, "spect")
            animator_call = irw.animator(config.dirFilter, "filter")
            animator_call = irw.animator(config.dirRecon, "recon")
            # show option
            # ishow= imgSHOW2(numberBlock)
            print('result show! To exit close picture!')
        plt.show()

    def process_detection(self, img, blackout: bool = False):
         Generates blocks in the set size with transformation and reconstruction with dct / idct.
            During the transformation steps the blocks will be put through a (object) detection.

        @param img: Bitwise image vector.
        @param blackout: Determine whether a block with no detected object in it should be blacked out.
        

        a = np.arange(0, self.number_coefficients)
        b = a.reshape(self.number_block, self.number_block)
        img2 = np.zeros_like(b).astype(float)
        imgBLOCK = timproc.imgBLOCK_tim(img, img2, self.number_block, self.animate, self.nbit, blackout)

    def blocks(self, img):
         Generates blocks in the set size.

        @param img: Bitwise image vector.
        

        #	from lib.blockprocess import imgBLOCK1
        # -------------------------------------
        #	os.mkdir("blocks/")

        a = np.arange(0, self.number_coefficients)
        b = a.reshape(self.number_block, self.number_block)
        img2 = np.zeros_like(b).astype(int)
        imgBLOCK = bp1.imgBLOCK1(img, img2, self.number_block, self.animate)
        animator_call = irw.animator(config.dirBlocks, "ori-blocks")

    def block_image(self, img):
         Generates blocks in the set size in the given image itself.

        @param img: Bitwise image vector.
        

        # Blocks im Bild
        # from lib.imgRW import imgSORT
        # imgBLOCK = imgSORT(img, numberBlock, 2)
        imgBLOCK = bp1.imgBlockinImage(img, self.number_block, self.animate)
        animator_call = irw.animator(config.dirBlocks, "ori-map")

    def video(self):
         Processing with live video feed

        print('*************** Live Cam  processing ****************')
        icall = lp.camera_CODEC(self.animate)
        sys.exit(0)

    def image_capture(self):
         Processing with an image that can be taken manually

        print('*************** ImageClick processing ****************')
        icall = lp.image_Click(self.animate)
        sys.exit(0)
"""
# -----------------------------------------------------------------------------
# Main Programm
# Class Codec
# Datum : 30.03.2021
# Authors: Dr.-Ing. The Anh Vuong 

def main():
    """ """

    # Parameter reading
    # Input Bilder BGR statt RGB
    print ('CODEC-KIT modus starten:' + '\n' + 'P=Process / Return, G=Blocks, M=BlockinImage,' \
    '\n' + 'V=Video, I=ImageCapture, D=Detection, A:Audio,T: Transform-BasisImages')

    number_modus = input("modus:" ) 
    
    #numberModus = input("modus (Return /P=Process2D, G=Blocks, M=BlockinImage, V=video, I=ImageCapture,A:auio/Process1D): ") 

    if number_modus == "":
        number_modus = "P"

    # ------------ Video Codec Processing  -----

    if number_modus in "VI":
        # Live feed video or image
        codec = board.Codec(0, 0)
        if number_modus == "V":
            codec.video()
        elif number_modus == "I":
            codec.image_capture()

    # ------------ Audio Codec Processing  -----
    
    if number_modus  == "A":
        codec = board.Codec(0, 0)
        codec.audio(config.audioToRead)
    
    #    icall = audioproc.plot(config.audioToRead)
    #    icall = audioproc.play(config.audioToRead)
    #    sys.exit(0)
    

    # ------------ Basis Bilder Generator  -----
    if number_modus  == "T":
        number_block = input("Blocklength: ")
        number_block = int(number_block)
        transform_typ = 1
        codec = board.Codec(0, 0)
        codec.process_basisgenerator2(number_block, transform_typ)
        
        sys.exit(0)
    if number_modus == "T1":
        number_block = input("Blocklength: ")
        number_block = int(number_block)
        transform_typ = 1
        codec = board.Codec(0, 0)
        img=[]
        codec.process_basisgenerator(img,number_block,config.dirGenerator,transform_typ)
        sys.exit(0)

    if number_modus  == "T2":
        # Read input parameter for stored image
        number_block = input("Blocklength: ")
        number_block = int(number_block)
        transform_typ = 1
        codec = board.Codec(0, 0)
        codec.bS_generator(number_block,transform_typ,config.dirGenerator)

        sys.exit(0)
    
    
    
    # ------------ Image Codec Processing  -----
    else:

        # Read input parameter for stored image
        number_block = input("Blocklength: ")
        number_block = int(number_block)

        if number_modus in "PD":
            nbit = input("2D LowpassFilter-Blocklength:")
            nbit = int(nbit)
        else:
            nbit = number_block

        codec = board.Codec(number_block, nbit)

        if codec.confirm(config.imageToRead):

            # Get input image
            print('*************** Input Datein lesen ****************')
            img = irw.imgC2G(config.imageToRead, config.imageGray, 1)
            height = img.shape[0]
            width = img.shape[1]
            print("pic height:" + str(height))
            print("pic width:" + str(width))

            # Start selected modus
            if number_modus == "P":
                codec.process(img)
            elif number_modus == "D":
                blackout_inp = input("Black out blocks with no detected object in them? (y /n): ")
                blackout = True if blackout_inp.lower() == "y" else False
                codec.process_detection(img, blackout)
            elif number_modus == "G":
                codec.blocks(img)
            elif number_modus == "M":
                codec.block_image(img)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
