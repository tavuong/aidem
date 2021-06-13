# Process_1D.py
# Object oriented - 1D Processing
# Datum : 18.04.2021
# Author Dr.-Ing. The Anh Vuong 
import cv2  # openCV
import numpy as np
import config
#from lib.codec_dct import *
import lib.dct as transf
import lib.idct as itransf

class Block1D:
	def __init__(self, nb,type):
		self.size = nb
		if type == 1:
			block = np.zeros((nb), dtype="int16")
		elif type == 2:
			block = np.zeros((nb), dtype=float)

		self.block = block

	def print(self,text):
	  	print("dc-KIT:" + text)
	
	def dct_1D(self):
		return transf.dct_1 (self.block)
	def idct_1D(self):
		return itransf.idct_1 (self.block)


"""
	def save_image(self,nf,name):
#	  	print("Filename is " + self.name)
		filename= name + str(nf)+ '.png'
		cv2.imwrite(filename,self.block)

	def save_image_jpg(self,nf,name):
#		print("Filename is " + self.name)
		filename= name + str(nf)+ '.jpg'
		cv2.imwrite(filename,self.block)

	def copy_block(self):
		return (self.block)

	def refill(self,miniblock,nbI,nbI1,nbK,nbK1):
		for i in range (nbI, nbI1):
			for k in range (nbK, nbK1):
				self.block[i][k]=miniblock[i-nbI][k-nbK]
#		return (self.block)

	def blackout_block(self):
		self.block = np.zeros_like(self.block).astype(float)

	def dct_2D(self):
		return dc.dct_2 (self.block)

	def idct_2D(self):
		return dc.idct_2 (self.block)

	def lowpass_2D(self,nbit):
		return dc.lowpass_2d (self.block,nbit)
"""

def createVektor(number_block):
		"""
		@ number_block: np float 2 Dim. Array 
		"""
		nb = number_block
		block = np.zeros((nb), dtype=float)
		return block
