"""
Title           :CompressionCallback.py
Description     :Custom callback for dynamic model compression
Author          :Ilke Cugu
Date Created    :19-02-2019
Date Modified   :15-05-2019
version         :1.5
python_version  :3.6.6
"""

import keras
from keras import backend as K
from layers.MicroConv2D import MicroConv2D

class CompressionCallback(keras.callbacks.Callback):
	def on_train_end(self, logs=None):
		# Trigger dynamic model compression
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				if K.get_value(layer.disable_compression) == 0:
					layer.kill_insignificants()
				else:
					layer.mask_dead_kernels()