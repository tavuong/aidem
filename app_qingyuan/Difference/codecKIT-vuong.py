# CodecKIT.py
# Frame Work für block-codierung 
# Status: Entwicklung
# Berechen Codierung eines Bilders aus JPEG mit OPencv
# Der Vorgang wird analyse
# Die Spektrum wird als Jpeg-Bilde rerzeugt!
# Die Matrix des Spektrum wird:
# - direckt in der Rechnung von Oroiginal -> Spectrum
# - indirect aus der reconstruierte Bild 
# Developer: Dr. -Ing. The Anh Vuong

# -*- coding: utf-8 -*-

import sys

import numpy as np

from bloc_process2 import *
from lib.bloc_process import *
from lib.imgRW import *

# Parameter reading
# Input Bilder BGR statt RGB
numberModus = input("modus (P=Process, G=Blocks, M=Blockmapping): ")
numberBlock = input("Blocklength: ")
numberBlock = int(numberBlock)
numberCoefficients = numberBlock * numberBlock

# for test param is set to 1
# param = input ("Blockstoring? Yes=1 / no =0: ") 
# param = int (param)
param = 1

nbit = input("Spect-Compress length:")
nbit = int(nbit)

print("image_input " + config.imageToRead + " !" + "\n")
print("Blocklength: " + str(numberBlock) + " !" + "\n")
print("Number of DCT-coefficients: " + str(numberCoefficients) + " !" + "\n")
print("Compress length: " + str(nbit) + " !" + "\n")
rcontinue = input("is setting correct (y /n)? ")
if rcontinue == "n": sys.exit(0)

print('*************** Input Datein lesen ****************')
img = imgC2G(config.imageToRead, config.imageGray, 1)
nb = numberBlock
height = img.shape[0]
width = img.shape[1]
print("pic height:" + str(height))
print("pic width:" + str(width))

print('*************** Blockdatei processing ****************')
# Mini Blocks generator 
#	from lib.blockprocess import imgBLOCK1 
# -------------------------------------
#	os.mkdir("blocks/")
if numberModus in "G":
    a = np.arange(0, numberCoefficients)
    b = a.reshape(numberBlock, numberBlock)
    img2 = np.zeros_like(b).astype(int)
    imgBLOCK = imgBLOCK1(img, img2, numberBlock, 2)
    animator_call = animator(config.dirBlocks, "ori-blocks")

# -------------------------------------
# Blocks im Bild
# from lib.imgRW import imgSORT
# imgBLOCK = imgSORT(img, numberBlock, 2)  
if numberModus in "M":
    imgBLOCK = imgSORT2(img, numberBlock, 2)
    animator_call = animator(config.dirBlocks, "ori-map")

# -------------------------------------
# Blocks generator
# dct  -> Spect- blocks
# odct -> reconstr -blocks 
# blocks --> Big image back
# from lib.blockprocess import imgBLOCK1 
# -------------------------------------

if (numberModus in "P"):
    a = np.arange(0, numberCoefficients)
    b = a.reshape(numberBlock, numberBlock)
    img2 = np.zeros_like(b).astype(float)
    imgBLOCK = imgBLOCK2(img, img2, numberBlock, param, nbit)
    print('*************** output ./img_out/movie*.gif ***********')
    if (param == 1):
        animator_call = animator(config.dirBlocks, "ori")
        animator_call = animator(config.dirSpect, "spec")
        animator_call = animator(config.dirRecon, "recon")
