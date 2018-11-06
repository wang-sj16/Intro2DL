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
		self.log = logit
		self.g = gt
		logit=np.exp(logit)
		for i in range(logit.shape[0]):
				logit[i]=logit[i]/np.sum(logit[i])
		self.loss=np.sum(np.log(logit)*gt)/(-batch_size * 10)
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
