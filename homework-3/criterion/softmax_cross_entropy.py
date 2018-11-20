""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 1)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.

		self.gt = gt
		self.prob = np.exp(logit) / (EPS + np.exp(logit).sum(axis=1, keepdims=True))

		# calculate the accuracy
		predict_y = np.argmax(self.prob, axis=1) # self.prob, not logit.
		gt_y = np.argmax(gt, axis=1)
		self.acc = np.mean(predict_y == gt_y)

		# calculate the loss
		loss = np.sum(-gt * np.log(self.prob + EPS), axis=1)
		self.loss = np.mean(loss)
	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		return self.prob - self.gt
	    ############################################################################
