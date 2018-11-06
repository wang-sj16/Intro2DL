""" Softmax Cross-Entropy Loss Layer """

import numpy as np
import math

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11
batch_size = 100

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')
		
	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """
		
		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		#self.loss = np.zeros(1, dtype='f')
		self.log = logit
		self.g = gt
		#denominator = [0.0 for i in range(batch_size)]
		#for i in range(batch_size):
		#	for j in range(10):
		#		denominator[i] += np.exp(logit[i][j]-max(logit[i]))
		#		
		#for i in range(batch_size):
		#	for j in range(10):
		#		self.log[i][j] = (np.exp(logit[i][j]-max(logit[i]))) / denominator[i]
		#		self.loss += math.log(max(self.log[i][j] , EPS)) * gt[i][j]
		#	self.acc += (np.argmax(logit[i])==np.argmax(gt[i]))
		#	
		#	
		#self.loss = -self.loss / (batch_size * 10)
		#self.acc = self.acc / (batch_size)
		self.loss=np.sum(np.log(logit+EPS)*gt)/(-batch_size * 10)
		self.acc=np.sum(np.argmax(logit,axis=1)-np.argmax(gt,axis=1)==0)/batch_size
		return self.loss
        ############################################################################


	def backward(self):
		self.log -= self.g
		return self.log
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
	    ############################################################################
