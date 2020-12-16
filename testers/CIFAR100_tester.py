"""
Title           :CIFAR100_tester.py
Description     :Easy CIFAR-100 benchmark
Author          :Ilke Cugu
Date Created    :17-10-2019
Date Modified   :02-05-2020
version         :1.2.7
python_version  :3.6.6
"""

import keras
from overrides import overrides
from applications.microresnet.MicroResNet import MicroResNet
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from testers.Base_tester import Base_tester

class CIFAR100_tester(Base_tester):
	def __init__(self, wait=False):
		"""
		Conditional constructor to enable creating testers without immediately triggering the dataset loader.

		# Arguments
			:param wait: (bool) If True, then the constructor will onyl create the instance and
								wait for manual activation to actually load the dataset
		"""
		super(CIFAR100_tester, self).__init__()
		if not wait:
			self.activate()

	def activate(self):
		"""
		Call this function to manually start dataset deployment
		"""
		# Deploy the dataset
		self.dataset = cifar100.load_data()
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset
		self.preprocess_dataset()
		self.dim_x = 32
		self.dim_y = 32

		# Split the train and validation sets
		self.training_set_size = int(self.x_train.shape[0] * 0.9)
		self.val_set_size = int(self.x_train.shape[0] * 0.1)

		# Turn the labels into one-hot-vector form
		self.y_train = keras.utils.to_categorical(self.y_train, self.get_n_classes())
		self.y_test = keras.utils.to_categorical(self.y_test, self.get_n_classes())

	def createImgGenerator(self):
		return ImageDataGenerator(
			# set input mean to 0 over the dataset
			featurewise_center=False,
			# set each sample mean to 0
			samplewise_center=False,
			# divide inputs by std of dataset
			featurewise_std_normalization=False,
			# divide each input by its std
			samplewise_std_normalization=False,
			# apply ZCA whitening
			zca_whitening=False,
			# epsilon for ZCA whitening
			zca_epsilon=1e-06,
			# randomly rotate images in the range (deg 0 to 180)
			rotation_range=0,
			# randomly shift images horizontally
			width_shift_range=0.1,
			# randomly shift images vertically
			height_shift_range=0.1,
			# set range for random shear
			shear_range=0.,
			# set range for random zoom
			zoom_range=0.,
			# set range for random channel shifts
			channel_shift_range=0.,
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			# value used for fill_mode = "constant"
			cval=0.,
			# randomly flip images
			horizontal_flip=True,
			# randomly flip images
			vertical_flip=False,
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.1
		)

	def lr_schedule(self, epoch):
		"""
		The same learning rate scheduler for CIFAR-10 as part of callbacks during training

		Adopted from https://keras.io/examples/cifar10_resnet/

		# Arguments
			:param epoch: (int) # of epochs

		:return: new learning rate
		"""
		lr = self.base_lr
		if epoch > 180:
			lr *= 0.5e-3
		elif epoch > 160:
			lr *= 1e-3
		elif epoch > 120:
			lr *= 1e-2
		elif epoch > 80:
			lr *= 1e-1
		return lr

	@overrides
	def get_n_classes(self):
		return 100

	@overrides
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
		# CIFAR-100 Specific configurations
		self.base_lr = lr
		new_callbacks = [] if callbacks is None else callbacks
		if schedule_lr and custom_lr_scheduler is None:
			lr_scheduler = LearningRateScheduler(self.lr_schedule, verbose=verbose)
			new_callbacks.extend([lr_scheduler])
		elif schedule_lr and custom_lr_scheduler is not None:
			new_callbacks.append(custom_lr_scheduler)

		# Compile the model
		optimizer = self.get_optimizer(optimizer, lr=lr, momentum=momentum, decay=decay)
		model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

		# Training data preparation
		trainImgGenerator = self.createImgGenerator()
		trainImgGenerator.fit(self.x_train)
		trainGenerator = trainImgGenerator.flow(self.x_train, self.y_train, subset="training", batch_size=batch_size, shuffle=True, seed=13)
		valGenerator = trainImgGenerator.flow(self.x_train, self.y_train, subset="validation", batch_size=batch_size, shuffle=False)

		# Train the model
		_model = model.model if isinstance(model, MicroResNet) else model
		hist = _model.fit_generator(
			trainGenerator,
			epochs=epochs,
			verbose=verbose,
			steps_per_epoch=self.training_set_size // batch_size,
			validation_data=valGenerator,
			validation_steps= self.val_set_size // batch_size,
			class_weight='auto',
			callbacks=new_callbacks
		)

		# Evaluate the model with the test data
		score = _model.evaluate(self.x_test, self.y_test, verbose=verbose)

		return hist, score