"""
Title           :Base_tester.py
Description     :Base class for dataset benchmarks
Author          :Ilke Cugu
Date Created    :16-01-2020
Date Modified   :02-05-2020
version         :1.0.2
python_version  :3.6.6
"""

import keras
import numpy as np

class Base_tester:
	def __init__(self, wait=False):
		# Base class variables
		self.base_lr = 1e-3
		self.dataset = None
		self.x_train = None
		self.y_train = None
		self.x_val = None
		self.y_val = None
		self.x_test = None
		self.y_test = None
		self.dim_x = None
		self.dim_y = None
		self.training_set_size = None
		self.val_set_size = None
		self.test_set_size = None

	def preprocess_dataset(self):
		# Normalize the data
		self.x_train = self.x_train.astype('float32') / 255.0
		self.x_test = self.x_test.astype('float32') / 255.0

		x_train_mean = np.mean(self.x_train, axis=0)
		self.x_train -= x_train_mean
		self.x_test -= x_train_mean

	def get_optimizer(self, optimizer, lr=None, momentum=None, decay=None):
		if optimizer == "adam":
			return keras.optimizers.Adam(lr=lr)
		elif optimizer == "sgd":
			return keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay)
		elif optimizer == "rmsprop":
			return keras.optimizers.RMSprop(lr=lr, momentum=momentum, decay=decay)
		elif optimizer == "adagrad":
			return keras.optimizers.Adagrad(lr=lr, momentum=momentum, decay=decay)
		else:
			return None

	def get_n_classes(self):
		return None

	def get_input_shape(self):
		return self.x_train.shape[1:]

	def get_y_test(self):
		return self.y_test

	def evaluate(self, model):
		return model.evaluate(self.x_test, self.y_test, verbose=0)

	def predict(self, model):
		return model.predict(self.x_test, verbose=0)

	def run(self, model,
			optimizer="adam",
			lr=1e-3,
			momentum=None,
			decay=None,
			loss='categorical_crossentropy',
			batch_size=128,
			epochs=200,
			verbose=0,
			callbacks=None,
			schedule_lr=True,
			custom_lr_scheduler=None):
		"""
		Runs the benchmark

		# Arguments
			:param model: Keras model (including MicroResNet)
			:param optimizer: (string) name of the selected Keras optimizer
			:param lr: (float) learning rate
			:param momentum: (float) only relevant for the optimization algorithms that use momentum
			:param decay: (float) only relevant for the optimization algorithms that use weight decay
			:param loss: (string) name of the selected Keras loss function
			:param batch_size: (int) # of inputs in a mini-batch
			:param epochs: (int) # of full training passes
			:param verbose: (int) Keras verbose argument
			:param callbacks: list of Keras callbacks
			:param schedule_lr: (bool) enable/disable learning rate scheduler (default or custom)
			:param custom_lr_scheduler: user defined learning rate scheduler

		:return: (history, score)
		"""
		hist = None
		score = None

		return hist, score