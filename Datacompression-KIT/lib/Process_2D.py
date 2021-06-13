# Process_2D.py
# Object oriented - 2D Processing
# Datum : 30.03.2021
# Author Dr.-Ing. The Anh Vuong 
import cv2  # openCV
import numpy as np

import config
#from lib.codec_dct import *
import lib.codec_dct as dc

class Block2D:
	def __init__(self, block):
		self.block = block
	
	def create(self,number_block):
		"""
		Create nummpy Array [nunber_block, number Block], type =float
		@ number_block
		"""
		nb = number_block
		self.block = np.zeros((nb,nb), dtype=float)

		return self.block


	def print(self,text):
	  	print("dc-KIT:" + text)

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

def createArray(number_block):
		"""
		Create nummpy Array [nunber_block, number Block], type =float
		@ number_block
		"""
		nb = number_block
		block = np.zeros((nb,nb), dtype=float)

		return block
