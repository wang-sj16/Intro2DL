# -*- encoding: utf-8 -*-

import numpy as np
batch_size=100
class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		self.Input = Input
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		out_len=(Input.shape[3]+2*self.pad)//self.kernel_size
		out_wid=(Input.shape[3]+2*self.pad)//self.kernel_size        
		output=np.zeros((Input.shape[0],Input.shape[1],out_len,out_wid))
		self.flag=np.zeros(input_after_pad.shape)
		#print(self.flag.shape)
		for i in range(0,Input.shape[0]):
			for j in range(0,Input.shape[1]):
				pic=input_after_pad[i][j]
				row=0
				col=0
				for m in range(out_len):
					col=0
					for n in range(out_wid):
						output[i][j][m][n]=np.max(pic[row:row+self.kernel_size,col:col+self.kernel_size])
						x=np.argmax(pic[row:row+self.kernel_size,col:col+self.kernel_size])//self.kernel_size
						y=np.argmax(pic[row:row+self.kernel_size,col:col+self.kernel_size])% self.kernel_size
						x+=self.kernel_size*m
						y+=self.kernel_size*n
						self.flag[i][j][x][y]=1
						col+=self.kernel_size
					row+=self.kernel_size
		return output
		############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		kernel_size=self.kernel_size
		#h=delta.shape[2]
		#w=delta.shape[3]
		output=np.zeros((delta.shape[0],delta.shape[1],delta.shape[2]*self.kernel_size,delta.shape[3]*self.kernel_size))
		#output=np.zeros(self.flag.shape)
		for i in range(0,output.shape[0]):
			for j in range(0,output.shape[1]):
				pic=output[i][j]
				for m in range(pic.shape[0]):
					for n in range(pic.shape[1]):
						output[i][j][m][n]=delta[i][j][m//self.kernel_size][n//self.kernel_size]                    
		output=output*self.flag  
		return output
	    ############################################################################
