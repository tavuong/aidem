import lib_1D.audio_process as audioproc
import lib.dct as dct
import lib.idct as idct
import lib.bloc_process as bp1
import lib.bloc_process2 as bp2
import lib.live_process as lp
import lib.imgRW as irw
import lib.tim_process as timproc
import lib.basis_image as bigen
import lib.Process_2D as P2D
from math import cos, pi, sqrt
import numpy as np
import matplotlib.pylab as plt
import sys
import config
import cv2
"""
# Mangement for different Process
# Class Codec
# Datum : 12.04.2021
# Authors: Tim Orius, Dr.-Ing. The Anh Vuong 
"""

 
class Codec:

    def __init__(self, number_block, nbit, animate: int = 1):
        """Modus selection and preparation for codec

         @param number_block: length (pixel) per block.
         @param nbit: determine how large the filter should be.
         @param animate: determine whether the output should be animated.
         """

        self.animate = animate
        self.number_block = number_block
        self.nbit = nbit
        self.number_coefficients = number_block * number_block

    def confirm(self,image):
        """Confirmation of the initialization parameters

         @return confirmation input (True / False)
         """

#        print("image_input " + config.imageToRead + " !" + "\n")
        print("image_input " + image + " !" + "\n")
        print("Blocklength: " + str(self.number_block) + " !" + "\n")
        print("Number of DCT-coefficients: " + str(self.number_coefficients) + " !" + "\n")
        print("2D LowpassFilter- Blocklength: " + str(self.nbit) + " !" + "\n")
        rcontinue = input("is setting correct (y /n)? ")
        return True if (rcontinue != "n") else False

    def process(self, img):
        """ Generates blocks in the set size with transformation and reconstruction with dct / idct.

        @param img: Bitwise image vector.
        """

        # -------------------------------------
        # Blocks generator
        # dct  -> Spect- blocks
        # odct -> reconstr -blocks
        # blocks --> Big image back
        # from lib.blockprocess import imgBLOCK1
        # -------------------------------------

#        a = np.arange(0, self.number_coefficients)
#        b = a.reshape(self.number_block, self.number_block)
        b = P2D.createArray(self.number_block)
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

    def process_detection(self, img, blackout: bool = False, black_blocks: int = 0):
        """ Generates blocks in the set size with transformation and reconstruction with dct / idct.
            During the transformation steps the blocks will be put through a (object) detection.

        @param img: Bitwise image vector.
        @param blackout: Determine whether a block with no detected object in it should be blacked out.
        """

        a = np.arange(0, self.number_coefficients)
        b = a.reshape(self.number_block, self.number_block)
        img2 = np.zeros_like(b).astype(float)
        imgBLOCK = timproc.imgBLOCK_tim(img, img2, self.number_block, self.animate, self.nbit, blackout, black_blocks)

    def blocks(self, img):
        """ Generates blocks in the set size.

        @param img: Bitwise image vector.
        """

        #	from lib.blockprocess import imgBLOCK1
        # -------------------------------------
        #	os.mkdir("blocks/")

        a = np.arange(0, self.number_coefficients)
        b = a.reshape(self.number_block, self.number_block)
        img2 = np.zeros_like(b).astype(int)
        imgBLOCK = bp1.imgBLOCK1(img, img2, self.number_block, self.animate)
        animator_call = irw.animator(config.dirBlocks, "ori-blocks")

    def block_image(self, img):
        """ Generates blocks in the set size in the given image itself.

        @param img: Bitwise image vector.
        """

        # Blocks im Bild
        # from lib.imgRW import imgSORT
        # imgBLOCK = imgSORT(img, numberBlock, 2)
        imgBLOCK = bp1.imgBlockinImage(img, self.number_block, self.animate)
        animator_call = irw.animator(config.dirBlocks, "ori-map")

    def video(self):
        """ Processing with live video feed"""

        print('*************** Live Cam  processing ****************')
        icall = lp.camera_CODEC(self.animate)
        sys.exit(0)

    def image_capture(self):
        """ Processing with an image that can be taken manually"""

        print('*************** ImageClick processing ****************')
        icall = lp.image_Click(self.animate)
        sys.exit(0)
    
    def audio (self, file_name):
        """ Processing with Audio """

        print('*************** Audio  processing ****************')
        icall = audioproc.plot(file_name)
        icall = audioproc.play(file_name)
        sys.exit(0)
    
    # Modus T1: Basis Image calculation old structur
    def bS_generator(self,number_block,transform_typ,dirname):
        basis = bigen.DCT_Basis_Image()
        basis.size = number_block
        basis.folderName = dirname
        print(number_block , dirname) 
        if transform_typ == 1:
            basis.create()
        sys.exit(0)
    
    # Modus T: Basis Image Name after Matrix-Position of spectrals

    def process_basisgenerator(self,img,number_block,dirname,transform_typ):
        nb = number_block
        basis_Img= P2D.createArray(nb)
        basis = bigen.DCT_Basis_Image()
        basis.size = nb
        basis.folderName = dirname
        for u in range (0,nb):
            for v in range (0,nb):
                basis_Img = basis.create_NM (transform_typ, u,v)
                file_name = dirname + str(u)+ str(v) + ".jpg"
                print ("Basis Image: "+ file_name)
                cv2.imwrite(file_name,basis_Img)
        sys.exit(0)

    def process_basisgenerator2(self,number_block, transform_typ):
        bigen.basImgBLOCK2(number_block, transform_typ)
        sys.exit(0)

