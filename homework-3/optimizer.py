"""Optimizer Class"""

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:
				# TODO: Put your code here
				# Calculate diff_W and diff_b
				layer.diff_W = -self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
				layer.diff_b = -self.learningRate * layer.grad_b
				# Weight update
				layer.W += layer.diff_W
				layer.b += layer.diff_b


class SGDwithMomentum():
	def __init__(self, learningRate, weightDecay, momentum):
		self.learningRate = learningRate
		self.weightDecay = weightDecay
		self.momentum = momentum

	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:
				# TODO: Calculate diff_W and diff_b with momentum
				pass
				# Weight updating
				layer.W += layer.diff_W
				layer.b += layer.diff_b

