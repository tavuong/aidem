"""
    Author: Kelvin Tsang
    Github: https://github.com/Kelviiiin/DCT_Basis_Image

    Creates the basis images of discrete cosine tranformation.
    https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png
    
    Modify: Dr. The Anh Vuong, 17.04.2021

"""

import os
from math import cos, pi
import cv2
import numpy as np
import lib.Process_2D as P2D
import lib.imgRW as irw
import config

#  Modified for Datcompression-KIT: Dr. The Anh Vuong 
#  folderName over config.dirGenerator
class DCT_Basis_Image:
    def __init__(self):
        # size of the square image 8x8
        self.size = 1
        # folder name where the images are saved
        self.folderName = ""

    def create(self):
        # create image set
        imageSet = []
        for v in range(0, self.size):
            for u in range(0, self.size):
                basisImg = self.getBasisImage(u, v)
                imageSet.append(basisImg)

        # create folder
 #       if (not os.path.isdir(self.folderName)):
 #           os.makedirs(self.folderName)

        # name of the images are numbered consecutively
        ifull = len (imageSet)
        for i in range(0, len(imageSet)):
            number = None
            if(i < 10):
                number = "0" + str(i)
            else:
                number = str(i)
            # image directory
            directory = self.folderName + number + '.png'
            # save images with cv2
            cv2.imwrite(directory, imageSet[i])

        print('Done')
        return ifull

    # discrete cosinus transform
    def dct(self, x, y, u, v, n):
        return cos(((2 * x + 1) * (u * pi)) / (2 * n)) * cos(((2 * y + 1) * (v * pi)) / (2 * n))

    def cosValueToGrayValue(self, val):
        return int((val+1)/2 * 256)

    def getBasisImage(self, u, v):
        # for a given (u,v), make a DCT basis image
        basisImg = np.zeros((self.size, self.size))
        for y in range(0, self.size):
            for x in range(0, self.size):
                basisImg[y, x] = self.cosValueToGrayValue(self.dct(x, y, u, v, self.size))
        return basisImg
    
    def create_NM (self, transform_typ,ui,uj):
        """
        Generation Basis Image in Ordnung NM 
        @ BasisImage ui, uj , e.g. Baais Image 23
        @ trasnform_typ : 1 for DCT
        @ Author: Dr. The Anh Vuong
        """
        #   transform_typ = 1
        number_block = self.size
        basis = np.zeros((self.size, self.size))
        basis = self.getBasisImage(ui,uj)        
        return basis

"""
    Generation Basis Image in Ordnung NM 
    @ BasisImage ui, uj , e.g. Baais Image 23
    @ trasnform_typ : 1 for DCT
    @ output alle Baisbilde in einem Gross Bild
    @ 19.04.2021
    @ Author: Dr. The Anh Vuong
"""

def basImgBLOCK2(numberBlock=0, transform_typ=0):
	
	nb = numberBlock
	transform_typ =1
	height = nb*nb
	nzahl = int(height / nb)

#   ------------ Preset full-Image ---------------
	imgReconful = P2D.createArray(height)
	precont = P2D.Block2D(imgReconful)

	ncof = 0
	ncof = int(nb * nb)
	nf = 0
	nbK=0
	nbK1=nb
	nbI=0
	nbI1=nb
#   ------------ Preset Mini-Block --------------------- 

	basis_Img= P2D.createArray(nb)	
	basis = DCT_Basis_Image()
	basis.size = nb
	basis.folderName = config.dirGenerator # "./blocks_gen/"

#   ------------ Pipe Processing --------------------- 
#
# Blockweise processing implimented
# 
	for ri in range (nzahl):
		for r in range (nzahl):
			basis_Img = basis.create_NM (transform_typ,ri,r)
			file_name = basis.folderName + str(ri)+ str(r) + ".jpg"
			print ("Basis Image: "+ file_name)
			cv2.imwrite(file_name,basis_Img)

#---nf = Block index 
			nf = nf  +1
#--------------------Block Fill to Image -----------------------------
			precont.refill (basis_Img,nbI,nbI1,nbK,nbK1)
			nbK = nbK + nb
			nbK1= nbK1 + nb
		nbI = nbI + nb
		nbI1= nbI1 + nb
		nbK=0
		nbK1=nb	

# ---- Save full immages
	filename_save = config.dirImgout + '/Basis-'
#	ps = P2D.Block2D(imgReconful)
	precont.save_image_jpg(nf,filename_save)
	
	filename= filename_save + str(nf)+ '.jpg'

	ishow= irw.imgSHOW(filename)
	return