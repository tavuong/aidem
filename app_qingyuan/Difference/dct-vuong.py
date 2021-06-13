# VUONG-DCT.py
# Frame Work für DCT Bilder
# Status: Entwicklung
# Berechen DCT eines Bilders aus JPEG mit OPencv
# Der Vorgang wird analyse
# Die Spektrum wird als Jpeg-Bilde rerzeugt!
# Die Matrix des Spektrum wird:
# - direckt in der Rechnung von Oroiginal -> Spectrum
# - indirect aus der reconstruierte Bild 
# Developer: Dr. -Ing. The Anh Vuong

# -*- coding: utf-8 -*-

import cv2  # openCV
from config import *
from lib.imgRW import *
from lib.dct import *
from lib.idct import *
from math import cos, pi, sqrt
import numpy as np


def main():
	# Parameter reading
	# Input Bilder BGR statt RGB
	numberBlock = input("Blocklength: ") 
	numberBlock = int(numberBlock)
	numberCoefficients = numberBlock * numberBlock
	print("Number of DCT-coefficients" + str(numberCoefficients) + " !" + "\n")

	print('*************** Input Datein lesen ****************')
#	img = cv2.imread(config.imageToRead)
	img = imgREAD(config.imageToRead, 1)   

	print('*************** DCT_Sprechtral Berechnen ****************')
	imgDCT = dct_2d(img,numberCoefficients)
#	cv2.imwrite(config.imageSpec,imgDCT)
	imgDCT = imgWRITE(config.imageSpec,imgDCT,0)
	
	print('*************** Reconstruierte Bild  berechnen ****************')
	imgIDCT = idct_2d(imgDCT)
	# zu jpg schreiben
	# cv2.imwrite(config.imageReconstruct, idct_img) 
	imgIDCT = imgWRITE(config.imageReconstruct,imgIDCT,1)

	print('*************** Spectrum aus Reconst_Berechnen zum Test ****************')
	# Matrix Daten aus dem REC-JPEG berechnen
	# Reconstruct Bilder lesen
#	img = cv2.imread(config.imageReconstruct)
#   Einssuaschlaten
#	img = imgREAD(config.imageReconstruct, 0)   

	# Sprechtral Matrix neu rechnen	
#	imgDCT = dct_2d(img,numberCoefficients)
#	cv2.imwrite(config.imageSpec2,imgDCT)
#	imgDCT= imgWRITE(config.imageSpec2,imgDCT,0)
#	test = imgSHOW(config.imageSpec2)
	

main()

