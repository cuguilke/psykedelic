"""
Title           :tinyimagenet_tester.py
Description     :Easy Tiny ImageNet benchmark
Author          :Ilke Cugu
Date Created    :17-06-2019
Date Modified   :02-05-2020
version         :1.4.1
python_version  :3.6.6
"""
import os
import keras
import numpy as np
from overrides import overrides
from keras.utils.data_utils import get_file
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from applications.microresnet.MicroResNet import MicroResNet
from sklearn.model_selection import StratifiedShuffleSplit
from testers.Base_tester import Base_tester

class tinyimagenet_tester(Base_tester):
	def __init__(self, wait=False):
		"""
		Conditional constructor to enable creating testers without immediately triggering the dataset loader.

		# Arguments
			:param wait: (bool) If True, then the constructor will onyl create the instance and
								wait for manual activation to actually load the dataset
		"""
		super(tinyimagenet_tester, self).__init__()
		self.dataset_path = None
		self.x_train_mean = None
		self.label_dict = {}

		if not wait:
			self.activate()

	def activate(self):
		"""
		Call this function to manually start dataset deployment
		"""
		# Deploy the dataset
		self.dim_x = 64
		self.dim_y = 64
		self.dataset = self.load_data()
		(self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = self.dataset
		self.preprocess_dataset()

		# Split the train and validation sets
		self.training_set_size = int(self.x_train.shape[0])
		self.val_set_size = int(self.x_val.shape[0])
		self.test_set_size = int(self.x_test.shape[0])

		# Turn the labels into one-hot-vector form
		self.y_train = keras.utils.to_categorical(self.y_train, self.get_n_classes())
		self.y_val = keras.utils.to_categorical(self.y_val, self.get_n_classes())
		self.y_test = keras.utils.to_categorical(self.y_test, self.get_n_classes())

	@overrides
	def preprocess_dataset(self):
		# Normalize the data
		self.x_train = self.x_train.astype('float32') / 255.0
		self.x_val = self.x_val.astype('float32') / 255.0
		self.x_test = self.x_test.astype('float32') / 255.0

		self.x_train_mean = np.mean(self.x_train, axis=0)
		self.x_train -= self.x_train_mean
		self.x_val -= self.x_train_mean
		self.x_test -= self.x_train_mean

	def load_data(self):
		"""
		Loads tiny-imagenet dataset.

		:return: tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
		"""
		dirname = 'tiny-imagenet-200.zip'
		origin = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
		path = get_file(dirname, origin=origin, extract=True, archive_format="zip")
		path = path.split(".zip")[0]
		self.dataset_path = path

		num_train_samples = 100000
		num_test_samples = 10000

		x = np.empty((num_train_samples, self.dim_x, self.dim_y, 3), dtype='uint8')
		y = np.empty((num_train_samples,), dtype='uint8')

		x_test = np.empty((num_test_samples, self.dim_x, self.dim_y, 3), dtype='uint8')
		y_test = np.empty((num_test_samples,), dtype='uint8')

		# Deploy labels
		self.read_labels(path)
		reverse_label_dict = {v: k for k,v in self.label_dict.items()}

		# Gather training data
		counter = 0
		fpath = os.path.join(path, "train")
		for i in range(len(self.label_dict)):
			dir_path = os.path.join(fpath, self.label_dict[i], "images")
			files = os.listdir(dir_path)
			for file in files:
				if file.endswith("JPEG") or file.endswith("jpeg"):
					img_path = os.path.join(dir_path, file)
					img = img_to_array(load_img(img_path))
					x[counter, :, :, :] = img
					y[counter] = i
					counter += 1

		# Then split that into train/val sets
		train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=13)
		for i_train, i_val in train_val_split.split(x, y):
			x_train, x_val = x[i_train], x[i_val]
			y_train, y_val = y[i_train], y[i_val]

		# Gather the original validation data as the new test set
		counter = 0
		fpath = os.path.join(path, "val")
		label_file_path = os.path.join(fpath, "val_annotations.txt")
		with open(label_file_path, "r") as label_file:
			for line in label_file:
				temp = line[:-1].split("\t")
				img_path = os.path.join(fpath, "images", temp[0])
				img = img_to_array(load_img(img_path))
				label = reverse_label_dict[temp[1]]
				x_test[counter, :, :, :] = img
				y_test[counter] = label
				counter += 1

		return (x_train, y_train), (x_val, y_val), (x_test, y_test)

	def load_test_data(self):
		x_test = np.empty((10000, self.dim_x, self.dim_y, 3), dtype='uint8')

		# Gather test data
		counter = 0
		test_img_names = []
		fpath = os.path.join(self.dataset_path, "test", "images")
		files = os.listdir(fpath)
		for file in files:
			if file.endswith("JPEG") or file.endswith("jpeg"):
				img_path = os.path.join(fpath, file)
				img = img_to_array(load_img(img_path))
				test_img_names.append(file)
				x_test[counter, :, :, :] = img
				counter += 1

		# Preprocess the test set
		x_test = x_test.astype('float32') / 255.0
		x_test -= self.x_train_mean

		return x_test, test_img_names

	def read_labels(self, path):
		fpath = os.path.join(path, "wnids.txt")
		with open(fpath, "r") as label_file:
			i = 0
			for line in label_file:
				self.label_dict[i] = line[:-1]
				i += 1

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
			data_format="channels_last"
		)

	def lr_schedule(self, epoch):
		"""
		Learning rate scheduler for CIFAR-10 as part of callbacks during training

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
		return 200

	@overrides
	def evaluate(self, model):
		return model.evaluate(self.x_test, self.y_test, verbose=0)

	@overrides
	def predict(self, model):
		return model.predict(self.x_test, verbose=0)

	def print_predictions(self, model_name, preds, img_names):
		file_name = "%s_preds.txt" % model_name
		with open(file_name, "w+") as preds_file:
			for i in range(len(img_names)):
				entry = "%s %s\n" % (img_names[i], np.argmax(preds[i]))
				preds_file.write(entry)

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
			custom_lr_scheduler=None,
			print_preds_for_test_set=False):
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
			:param print_preds_for_test_set: (bool) enables printing out the test set predictions of the trained model

		:return: (history, score)
		"""
		# tiny-imagenet Specific configurations
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

		# Training & validation data preparation
		trainImgGenerator = self.createImgGenerator()
		trainGenerator = trainImgGenerator.flow(self.x_train, self.y_train, batch_size=batch_size, shuffle=True, seed=13)

		# Train the model
		_model = model.model if isinstance(model, MicroResNet) else model
		hist = _model.fit_generator(
			trainGenerator,
			epochs=epochs,
			verbose=verbose,
			steps_per_epoch=self.training_set_size // batch_size,
			validation_data=(self.x_val, self.y_val),
			class_weight='auto',
			callbacks=new_callbacks
		)

		# Evaluate the model with the val data
		score = _model.evaluate(self.x_test, self.y_test, verbose=verbose)

		# Print out the model predictions for the test data to upload the tiny-imagenet test server
		if print_preds_for_test_set:
			new_x_test, img_names = self.load_test_data()
			preds = model.predict(new_x_test, verbose=verbose)
			self.print_predictions(model.name, preds, img_names)

		return hist, score