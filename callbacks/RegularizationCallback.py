"""
Title           :RegularizationCallback.py
Description     :Callback for custom weight regularization
Author          :Ilke Cugu
Date Created    :23-03-2020
Date Modified   :11-05-2020
version         :1.1
python_version  :3.6.6
"""

import keras
import numpy as np
from keras import backend as K
from layers.MicroConv2D import MicroConv2D

class RegularizationCallback(keras.callbacks.Callback):
	def __init__(self, l1_penalty):
		super(RegularizationCallback, self).__init__()
		self.l1_penalty = l1_penalty

	def on_batch_end(self, batch, logs=None):
		# Revert L1 regularization on the diagonal of each kernel
		# The idea is to encourage large eigenvalues and small average weight per kernel
		revert_kernel = (np.ones((3,3)) - np.identity(3)) * self.l1_penalty * 100
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D) and layer.filter_size == 3:
				current_weights = layer.get_weights()
				kernel_shape = current_weights[0].shape

				for neuron in range(kernel_shape[3]):
					for depth in range(kernel_shape[2]):
						# Get the 3x3 convolution matrix
						sign = 1 if K.sum(current_weights[0][:, :, depth, neuron]) > 0 else -1
						current_weights[0][:, :, depth, neuron] += sign * revert_kernel

				layer.set_weights(current_weights)