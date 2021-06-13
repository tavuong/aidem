import cv2  # openCV
import config
from lib.codec_dct import dct_2, idct_2, coder
from lib.dct import dct_2d, dct_1d
from lib.idct import idct_2d, idct_1d
from PIL import Image
from math import cos, pi, sqrt
import numpy as np
# from scipy import fftpack
import os
import imageio


def imgBLOCK2(img, img2, numberBlock=0, param=0, nbit=0):
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
    #   ---------------------------------
    #   preset ordner
    #	os.mkdir("./blocks/")
    imageBlock = np.zeros_like(img2).astype(float)
    imageSpect = np.zeros_like(img2).astype(float)
    imageCoder = np.zeros_like(img2).astype(float)
    imageRecon = np.zeros_like(img2).astype(float)
    imgReconful = np.zeros_like(img).astype(float)
    imgSpectful = np.zeros_like(img).astype(float)

    ncof = 0
    ncof = int(nb * nb)
    nf = 0
    nbK = 0
    nbK1 = nb
    nbI = 0
    nbI1 = nb
    for ri in range(nzahl):
        for r in range(nzahl):
            for i in range(nbI, nbI1):
                for k in range(nbK, nbK1):
                    imageBlock[i - nbI][k - nbK] = img[i][k]
            print("Block calculated " + str(nf) + "\r")
            # Blockweise processing
            #			imageSpect= dct_2(imageBlock)
            imageCoder = dct_2(imageBlock)
            imageSpect = coder(imageCoder, nmap)
            imageRecont = idct_2(imageSpect)
            #			cv2.imwrite('./recon/recon' + str(nf)+ '.png',imageRecont)
            if (param == 1):
                cv2.imwrite('./blocks/block' + str(nf) + '.png', imageBlock)
                cv2.imwrite('./spect/spect' + str(nf) + '.png', imageSpect)
                cv2.imwrite('./recon/recon' + str(nf) + '.png', imageRecont)

            nf = nf + 1
            for i in range(nbI, nbI1):
                for k in range(nbK, nbK1):
                    imgReconful[i][k] = imageRecont[i - nbI][k - nbK]
                    imgSpectful[i][k] = imageSpect[i - nbI][k - nbK]

            nbK = nbK + nb
            nbK1 = nbK1 + nb
        nbI = nbI + nb
        nbI1 = nbI1 + nb
        nbK = 0
        nbK1 = nb
    cv2.imwrite("./img_out/dct_spec.jpg", imgSpectful)
    cv2.imwrite("./img_out/dct_reconst.jpg", imgReconful)
    return imageBlock
