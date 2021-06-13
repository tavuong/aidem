"""
    Author: Kelvin Tsang
    Github: https://github.com/Kelviiiin/DCT_Basis_Image

    Creates the basis images of discrete cosine tranformation.
    https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png
"""

import os
from math import cos, pi
import cv2
import numpy as np

class DCT_Basis_Image:
    def __init__(self):
        # size of the square image 8x8
        self.size = 8
        # folder name where the images are saved
        self.folderName = "BasisImage"

    def create(self):
        # create image set
        imageSet = []
        for v in range(0, self.size):
            for u in range(0, self.size):
                basisImg = self.getBasisImage(u, v)
                imageSet.append(basisImg)

        # create folder
        if (not os.path.isdir(self.folderName)):
            os.makedirs(self.folderName)

        # name of the images are numbered consecutively
        for i in range(0, len(imageSet)):
            number = None
            if(i < 10):
                number = "0" + str(i)
            else:
                number = str(i)
            # image directory
            directory = 'BasisImage/' + number + '.png'
            # save images with cv2
            cv2.imwrite(directory, imageSet[i])

        print('Done')

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


basisImage = DCT_Basis_Image()
basisImage.create()



