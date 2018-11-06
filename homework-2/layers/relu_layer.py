""" ReLU Layer """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.trainable = False # no parameters

	def forward(self, Input):
		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.
		self.input = Input
		return np.maximum(Input, 0)
	    ############################################################################


	def backward(self, delta):
		#for i in range(len(delta)):
			#for j in range(len(delta[i])):
				#delta[i][j] *= (self.inp[i][j] >= 0)
				
		return delta*(self.input>=0)
		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta


	    ############################################################################
