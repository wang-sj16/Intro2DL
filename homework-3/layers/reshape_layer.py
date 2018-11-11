"""Reshape layer"""

class ReshapeLayer():
	def __init__(self, input_shape, output_shape):
		"""
		Apply the reshape operation to the incoming data
		Args:
			num_input: size of each input sample
			num_output: size of each output sample
		"""
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.trainable = False

	def forward(self, Input):
		# TODO: put your code here

	def backward(self, delta):
		# TODO: put your code here
