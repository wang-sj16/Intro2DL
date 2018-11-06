""" Euclidean Loss Layer """

import numpy as np
batch_size = 100
EPS = 1e-11

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.
		
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
		self.grad = logit
		self.g = gt
		
		#logit -= gt
		#for i in range(batch_size):
		#	for j in range(10):
		#		self.loss += logit[i][j] * logit[i][j]	
		#	if (np.argmax(logit[i])==np.argmax(gt[i])):
		#		self.acc += 1
		self.loss = 0.5*np.sum((logit-gt)*(logit-gt))/ (batch_size * 10)
		#self.loss = self.loss * 0.5 / (batch_size * 10)
		#self.acc = self.acc / (batch_size)
		self.acc=np.sum(np.argmax(logit,axis=1)-np.argmax(gt,axis=1)==0)/batch_size
		return self.loss
	    ############################################################################

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		self.grad -= self.g	
		return self.grad
############################################################################
