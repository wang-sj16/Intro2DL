""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False
		
	def forward(self, Input):
		self.input = Input
		return  (1/ (1+ np.exp (-Input)))
		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
	    ############################################################################

	def backward(self, delta):
		#for i in range(len(delta)):
		#	for j in range(len(delta[i])):
		#		ne = np.exp(-self.inp[i][j])
		#		delta[i][j] *= (1.0 / (1.0 + ne)) * (1.0 - (1.0 / (1.0 + ne)))
		return delta*((1/(1+np.exp(-self.input)))*(1-1/(1+np.exp(-self.input))))
		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta


	    ############################################################################
