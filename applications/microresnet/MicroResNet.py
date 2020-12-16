"""
Title           :MicroResNet.py
Description     :Custom ResNet for dynamic model compression
Author          :Ilke Cugu
Date Created    :19-02-2019
Date Modified   :03-06-2020
version         :2.8.1
python_version  :3.6.6
"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.layers import InputLayer, Input, ZeroPadding2D, BatchNormalization, Activation, Flatten, Add
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from layers.MicroConv2D import MicroConv2D
from keras.regularizers import l1, l2, Regularizer
from keras_applications.resnet_common import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

class EigRegularizer(Regularizer):
	def __init__(self, l1=0., alpha=0., beta=0.):
		"""
		Custom regularization to test importance of eigenvalues vs. weights for pruning

		For 1x1 kernels, it is equal to the L1 regularizer

		# Arguments
			:param l1: (float) L1 regularization factor
			:param alpha: (float) weight of the diagonal elements
			:param beta: (float) weight of the off-diagonal elements
		"""
		self.l1 = K.cast_to_floatx(l1)
		self.alpha = K.cast_to_floatx(alpha)
		self.beta = K.cast_to_floatx(beta)

	def __call__(self, x):
		regularization = 0.
		if self.l1:
			if x.shape[0] > 1:
				diag = 0.
				for neuron in range(x.shape[3]):
					for depth in range(x.shape[2]):
						# Get the 3x3 convolution matrix
						kernel = x[:, :, depth, neuron]
						diag += K.sum(K.abs(tf.linalg.tensor_diag_part(kernel)))
				regularization += self.l1 * (self.alpha * diag + self.beta * (K.sum(K.abs(x)) - diag))
			else:
				regularization += self.l1 * K.sum(K.abs(x))
		return regularization

	def get_config(self):
		return {"l1": float(self.l1),
				"alpha": float(self.alpha),
				"beta": float(self.beta)}

class MicroResNet:
	"""
	    New ResNet implementation with dynamic model compression support
	"""
	def __init__(self, input_shape, n_classes, depth,
				 name=None,
				 disable_residual_connections=False,
				 disable_compression=False,
				 l1_penalty=None,
				 significance_threshold=None,
				 contribution_threshold=None,
				 compression_mode="det",
				 compression_rate=0.1,
				 pretrained_weights=None,
				 pretrained_MicroResNet=None):
		"""
		# Arguments
			:param input_shape: (3D tuple)
			:param n_classes: (int) # of classes
			:param depth: (int) defines the # of layers
				- {20, 32, 44, 56, 110}: thin ResNet constructor
				- {50, 101, 152}: full ResNet constructor which calls keras.applications.resnet functions
			:param name: (string) model name
			:param disable_residual_connections: (bool) removes the skip connections when enabled
			:param disable_compression: (bool) transforms microconv layers to standard conv layers
			:param l1_penalty: (float) penalty hyper-parameter for L1 regularization on MicroConv2D kernel weights
			:param significance_threshold: (float) threshold parameter for MicroConv2D layers
			:param contribution_threshold: (float) threshold parameter for MicroConv2D layers
			:param compression_mode: (string) hyper-parameter for MicroConv2D layers
			:param pretrained_weights: (.h5 file path or string) to use pre-trained model weights
			:param pretrained_MicroResNet: (keras model) to use pre-trained MicroResNet model
		"""
		self.name = name
		self.disable_residual_connections = disable_residual_connections
		self.disable_compression = disable_compression
		self.map_conv2microconv = {}
		self.l1_penalty = l1_penalty
		self.significance_threshold = significance_threshold
		self.contribution_threshold = contribution_threshold
		self.compression_mode = compression_mode
		self.compression_rate = compression_rate
		self.pretrained_weights = pretrained_weights
		self.model = None

		# Build the model
		if pretrained_MicroResNet is None:
			self.build(input_shape, n_classes, depth)
		else:
			self.deep_copy(pretrained_MicroResNet)

		# Rename the model
		if self.name is not None:
			self.model.name = self.name

	def build(self, input_shape, n_classes, depth):
		if depth in [50, 101, 152]:
			self.build_full(input_shape, n_classes, depth)
		elif (depth - 2) % 6 == 0:
			self.build_thin(input_shape, n_classes, depth)
		else:
			raise ValueError("Unsupported depth for MicroResNets!")

	def resnet_layer_thin(self, inputs,
						  num_filters=16,
						  kernel_size=(3, 3),
						  strides=1,
						  activation='relu',
						  batch_normalization=True,
						  conv_first=True):
		"""2D Convolution-Batch Normalization-Activation stack builder

		# Arguments
			:param inputs: (tensor) input tensor from input image or previous layer
			:param num_filters: (int) Conv2D number of filters
			:param kernel_size: (int tuple) Conv2D square kernel dimensions
			:param strides: (int) Conv2D square stride dimensions
			:param activation: (string) activation name
			:param batch_normalization: (bool) whether to include batch normalization
			:param conv_first: (bool) conv-bn-activation (True) or
				bn-activation-conv (False)

		# Returns
			x: (tensor) tensor as input to the next layer
		"""
		kernel_regularizer = l2(1e-4) if self.l1_penalty is None else l1(self.l1_penalty) #EigRegularizer(self.l1_penalty, alpha=0., beta=1.)
		conv = MicroConv2D(num_filters,
						   disable_compression=self.disable_compression,
						   significance_threshold=self.significance_threshold,
						   contribution_threshold=self.contribution_threshold,
						   compression_mode=self.compression_mode,
						   compression_rate=self.compression_rate,
						   kernel_size=kernel_size,
						   strides=strides,
						   padding='same',
						   kernel_initializer='he_normal',
						   kernel_regularizer=kernel_regularizer)

		x = inputs
		if conv_first:
			x = conv(x)
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
		else:
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
			x = conv(x)
		return x

	def build_thin(self, input_shape, n_classes, depth):
		"""
		This function generates a MicroResNet from thin ResNet models proposed by He et al for CIFAR-10

		Note: Codes are adopted from Chollet's ResNet_v1 implementation for CIFAR-10
		"""
		# Start model definition.
		num_filters = 16
		num_res_blocks = int((depth - 2) / 6)

		inputs = Input(shape=input_shape)
		x = self.resnet_layer_thin(inputs=inputs)
		if not self.disable_residual_connections:
			# Instantiate the stack of residual units
			for stack in range(3):
				for res_block in range(num_res_blocks):
					strides = 1
					if stack > 0 and res_block == 0:  # first layer but not first stack
						strides = 2  # downsample

					y = self.resnet_layer_thin(inputs=x, num_filters=num_filters, strides=strides)
					y = self.resnet_layer_thin(inputs=y, num_filters=num_filters, activation=None)

					if stack > 0 and res_block == 0:  # first layer but not first stack
						# Linear projection residual shortcut connection to match changed dims
						x = self.resnet_layer_thin(inputs=x, num_filters=num_filters, kernel_size=(1, 1), strides=strides, activation=None, batch_normalization=False)

					x = keras.layers.add([x, y])
					x = Activation('relu')(x)
				num_filters *= 2
		else:
			# If residual connections are disabled, do not add linear projection and merge layers
			for stack in range(3):
				for res_block in range(num_res_blocks):
					strides = 1
					if stack > 0 and res_block == 0:  # first layer but not first stack
						strides = 2  # downsample

					y = self.resnet_layer_thin(inputs=x, num_filters=num_filters, strides=strides)
					y = self.resnet_layer_thin(inputs=y, num_filters=num_filters, activation=None)
					x = Activation('relu')(y)
				num_filters *= 2

		# Add classifier on top.
		# The original model does not use BN after last shortcut connection-ReLU
		x = AveragePooling2D(pool_size=8)(x)
		y = Flatten()(x)
		outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(y)

		# Instantiate model.
		self.model = Model(inputs=inputs, outputs=outputs)

		# Load pretrained weights if applicable
		if self.pretrained_weights is not None and self.pretrained_weights.endswith(".h5"):
			self.model.load_weights(self.pretrained_weights)

	def resnet_layer(self, inputs,
					 num_filters,
					 kernel_size=3,
					 stride=1,
					 activation='relu',
					 conv_shortcut=True,
					 name=None):
		"""A residual block implementation taken from keras_applications.resnet_common

		# Arguments
			:param inputs: (tensor) input tensor.
			:param num_filters: (int) filters of the bottleneck layer.
			:param kernel_size: (int) default 3, kernel size of the bottleneck layer.
			:param stride: (int) default 1, stride of the first layer.
			:param activation: (string) activation name
			:param conv_shortcut: (bool) use convolution shortcut if True, otherwise identity shortcut.
			:param name: (string) block label.

		# Returns
			Output tensor for the residual block.
		"""
		epsilon = 1.001e-5
		bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
		kernel_regularizer = None if self.l1_penalty is None else l1(self.l1_penalty)

		x = inputs
		if conv_shortcut is True:
			shortcut = MicroConv2D(4 * num_filters,
								   name=name + '_0_microconv',
								   disable_compression=self.disable_compression,
								   significance_threshold=self.significance_threshold,
								   contribution_threshold=self.contribution_threshold,
								   compression_mode=self.compression_mode,
								   compression_rate=self.compression_rate,
								   kernel_size=1,
								   strides=stride,
								   kernel_initializer='he_normal',
								   kernel_regularizer=kernel_regularizer)(x)
			shortcut = BatchNormalization(axis=bn_axis, epsilon=epsilon, name=name + '_0_bn')(shortcut)
		else:
			shortcut = x

		x = MicroConv2D(num_filters,
						name=name + '_1_microconv',
						disable_compression=self.disable_compression,
						significance_threshold=self.significance_threshold,
						contribution_threshold=self.contribution_threshold,
						compression_mode=self.compression_mode,
						compression_rate=self.compression_rate,
						kernel_size=1,
						strides=stride,
						kernel_initializer='he_normal',
						kernel_regularizer=kernel_regularizer)(x)
		x = BatchNormalization(axis=bn_axis, epsilon=epsilon, name=name + '_1_bn')(x)
		x = Activation(activation, name=name + '_1_relu')(x)

		x = MicroConv2D(num_filters,
						name=name + '_2_microconv',
						disable_compression=self.disable_compression,
						significance_threshold=self.significance_threshold,
						contribution_threshold=self.contribution_threshold,
						compression_mode=self.compression_mode,
						compression_rate=self.compression_rate,
						kernel_size=kernel_size,
						padding='same',
						kernel_initializer='he_normal',
						kernel_regularizer=kernel_regularizer)(x)
		x = BatchNormalization(axis=bn_axis, epsilon=epsilon, name=name + '_2_bn')(x)
		x = Activation(activation, name=name + '_2_relu')(x)

		x = MicroConv2D(4 * num_filters,
						name=name + '_3_microconv',
						disable_compression=self.disable_compression,
						significance_threshold=self.significance_threshold,
						contribution_threshold=self.contribution_threshold,
						compression_mode=self.compression_mode,
						compression_rate=self.compression_rate,
						kernel_size=1,
						kernel_initializer='he_normal',
						kernel_regularizer=kernel_regularizer)(x)
		x = BatchNormalization(axis=bn_axis, epsilon=epsilon, name=name + '_3_bn')(x)

		if not self.disable_residual_connections:
			x = Add(name=name + '_add')([shortcut, x])
		x = Activation(activation, name=name + '_out')(x)

		return x

	def resnet_stack(self, inputs, num_filters, blocks, stride=2, name=None):
		x = inputs
		x = self.resnet_layer(x, num_filters, stride=stride, name=name + '_block1')
		for i in range(2, blocks + 1):
			x = self.resnet_layer(x, num_filters, conv_shortcut=False, name=name + '_block' + str(i))
		return x

	def micro_stack_fn(self, x, blocks_3, blocks_4):
		x = self.resnet_stack(x, 64, 3, stride=1, name='microconv2')
		x = self.resnet_stack(x, 128, blocks_3, name='microconv3')
		x = self.resnet_stack(x, 256, blocks_4, name='microconv4')
		x = self.resnet_stack(x, 512, 3, name='microconv5')
		return x

	def build_full(self, input_shape, n_classes, depth):
		"""
		This function generates a MicroResNet from a full ResNet models proposed by He et al

		Note: Codes are adopted from keras_applications.resnet_common
		"""
		bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

		blocks_3, blocks_4 = 0, 0
		if depth == 50:
			blocks_3, blocks_4 = 4, 6
		elif depth == 101:
			blocks_3, blocks_4 = 4, 23
		elif depth == 152:
			blocks_3, blocks_4 = 8, 36

		inputs = Input(shape=input_shape)
		x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
		x = Conv2D(64, 7, strides=2, name='conv1_conv', kernel_initializer='he_normal')(x)
		x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
		x = Activation('relu', name='conv1_relu')(x)

		x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
		x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

		x = self.micro_stack_fn(x, blocks_3, blocks_4)

		x = GlobalAveragePooling2D(name='avg_pool')(x)

		# Load ImageNet pre-trained weights
		if self.pretrained_weights == "imagenet":
			# Create base model = pretrained ResNet w/o top layer
			self.model = Model(inputs=inputs, outputs=x)

			# Load weights
			model_name = "resnet%s" % depth
			file_name = "%s_weights_tf_dim_ordering_tf_kernels_notop.h5" % model_name
			file_hash = WEIGHTS_HASHES[model_name][1]
			weights_path = keras.utils.get_file(file_name,
												BASE_WEIGHTS_PATH + file_name,
												cache_subdir='models',
												file_hash=file_hash)
			self.model.load_weights(weights_path)

			# Create the final model
			outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(self.model.layers[-1].output)
			self.model = Model(inputs=self.model.layers[0].input, outputs=outputs)

		else:
			# Create model
			outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
			self.model = Model(inputs=inputs, outputs=outputs)

		# ------------------------------------------------------ #
		# Old implementation to use ImageNet pretrained weights
		#self.build_iteratively(n_classes, baseModel)
		# ------------------------------------------------------ #

		# Load pretrained weights if applicable
		if self.pretrained_weights is not None and self.pretrained_weights.endswith("h5"):
			self.model.load_weights(self.pretrained_weights)

	def build_iteratively(self, n_classes, baseModel):
		"""
		@deprecated
		"""
		# Iterate through the base model and replace Conv2D layers with MicroConv2D layers
		input_layer = self.copy_layer(baseModel.layers[0])
		self.map_conv2microconv[baseModel.layers[0].name] = input_layer.output
		for i in range(1, len(baseModel.layers)):
			layer = baseModel.layers[i]
			if isinstance(layer, Conv2D):
				# Get the connection & config of the standard conv layer
				prev_layer = layer._inbound_nodes[0].inbound_layers[0]
				layer_config = layer.get_config()
				if self.l1_penalty is not None:
					layer_config['kernel_regularizer'] = l1(self.l1_penalty)

				# Create a new MicroConv2D layer with the original's connections & weights
				input = self.map_conv2microconv[prev_layer.name]
				microconv = MicroConv2D(pretrained_weights=layer.get_weights(),
										disable_compression=self.disable_compression,
										significance_threshold=self.significance_threshold,
										contribution_threshold=self.contribution_threshold,
										compression_mode=self.compression_mode,
										compression_rate=self.compression_rate,
										**layer_config)(input)

				# Save the output of the new layer in a dict to enable random connections across the model
				self.map_conv2microconv[layer.name] = microconv
			else:
				bridge = layer._inbound_nodes
				inputs = []

				# Iterate through prev layers to check if there is a microconv replacement
				for l in bridge[0].inbound_layers:
					inputs.append(self.map_conv2microconv[l.name])

				# If single connection, process straightforwardly
				if len(inputs) == 1:
					temp = self.copy_layer(layer, inputs[0])

				# If residual connections are disabled, remove the add layer
				elif self.disable_residual_connections:
					temp = inputs[0]

				# If residual connections are enabled, create a merge layer (e.g. Add layers in ResNet)
				else:
					temp = keras.layers.Add()(inputs)

				# Store the new connection
				self.map_conv2microconv[layer.name] = temp

		x = self.map_conv2microconv[baseModel.layers[len(baseModel.layers) - 1].name]
		preds = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)

		# Construct the new model
		self.model = Model(inputs=input_layer.input, outputs=preds)

	def neural_activity_check(self):
		result = True
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				result = result and layer.neural_activity_check()
		return result

	def report_compression_stats(self):
		report = []
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				report.append(layer.report_compression_stats())
		return report

	def copy_layer(self, layer, input=None):
		new_layer = layer
		layer_config = layer.get_config()

		if isinstance(layer, InputLayer):
			new_layer = InputLayer(**layer_config)
		elif isinstance(layer, ZeroPadding2D):
			new_layer = ZeroPadding2D(**layer_config)(input)
		elif isinstance(layer, BatchNormalization):
			new_layer = BatchNormalization(**layer_config)(input)
		elif isinstance(layer, Activation):
			new_layer = Activation(**layer_config)(input)
		elif isinstance(layer, MaxPooling2D):
			new_layer = MaxPooling2D(**layer_config)(input)
		elif isinstance(layer, GlobalMaxPooling2D):
			new_layer = GlobalMaxPooling2D(**layer_config)(input)

		return new_layer

	def deep_copy(self, pretrained_MicroResNet):
		self.model = pretrained_MicroResNet.model
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				layer.set_disable_compression(self.disable_compression)

	def freeze_weights(self):
		for layer in self.model.layers:
			layer.trainable = False

	def get_weights(self):
		return self.model.get_weights()

	def set_weights(self, weights):
		self.model.set_weights(weights)

	def get_compression_mode(self):
		return self.compression_mode

	def set_compression_mode(self, compression_mode):
		self.compression_mode = compression_mode
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				layer.set_compression_mode(compression_mode)

	def get_compression_ratio(self):
		return 1.0 - (self.count_params() / self.count_total_params())

	def compile(self, **kwargs):
		self.model.compile(**kwargs)

	def fit(self, x_train, y_train, **kwargs):
		return self.model.fit(x_train, y_train, **kwargs)

	def evaluate(self, x_test, y_test, **kwargs):
		return self.model.evaluate(x_test, y_test, **kwargs)

	def predict(self, x, **kwargs):
		return self.model.predict(x, **kwargs)

	def count_params(self):
		"""
		This function does not count the pruned weights
		:return:
		"""
		count = 0
		for layer in self.model.layers:
			count += layer.count_params()
		return count

	def count_total_params(self):
		"""
		This function also counts the pruned weights
		:return:
		"""
		count = 0
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				count += layer.get_total_param_count()
			else:
				count += layer.count_params()
		return count

	def plot_model(self, to_file="MicroResNet.png", show_shapes=True, show_layer_names=True):
		plot_model(self.model, to_file=to_file, show_shapes=show_shapes, show_layer_names=show_layer_names)

	def plot_compression_per_layer(self, to_file="MicroResNet_compression_per_layer.png"):
		# Chart configuration
		tableau20 = [(0, 70, 105), (0, 95, 135), (255, 127, 14), (255, 187, 120),
					 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
					 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
					 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
					 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
		bar_width = 0.35
		opacity = 0.8
		fig = plt.figure(figsize=(23, 11))

		# Process colors
		for i in range(len(tableau20)):
			r, g, b = tableau20[i]
			tableau20[i] = (r / 255., g / 255., b / 255.)

		# First, build active params vs. total params per layer chart
		ax1 = fig.add_subplot(2, 1, 1)
		layer_names = []
		total_params = []
		active_params = []
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				layer_names.append(layer.name)
				active_params.append(layer.count_params())
				total_params.append(layer.get_total_param_count())

		y_pos = np.arange(len(layer_names))
		ax1.bar(y_pos, active_params, bar_width, alpha=opacity, color=tableau20[6], label='Active', edgecolor=tableau20[6], hatch="")
		ax1.bar(y_pos + bar_width, total_params, bar_width, alpha=opacity, color=tableau20[1], label='Total', edgecolor=tableau20[0], hatch="//")

		ax1.set_xticks(y_pos + 0.5 * bar_width)
		ax1.set_xticklabels(layer_names, rotation=33)
		ax1.set_ylabel('# of params')
		ax1.legend(loc="upper left")
		ax1.grid(linestyle=":", linewidth=1, alpha=opacity)
		ax1.tick_params(axis="x", labelsize=9)

		# Second, build active/total params per layer chart
		ax2 = fig.add_subplot(2, 1, 2)
		active_param_percentage = []
		for i in range(len(total_params)):
			active_param_percentage.append((active_params[i]) * 100.0 / total_params[i])

		ax2.bar(y_pos, active_param_percentage, bar_width, alpha=opacity, color=tableau20[6], edgecolor=tableau20[6], hatch="//")

		ax2.set_xticks(y_pos)
		ax2.set_xticklabels(layer_names, rotation=33)
		ax2.set_ylabel('Active/Total param ratio (%)')
		ax2.set_ylim(ymin=0, ymax=100)
		ax2.grid(linestyle=":", linewidth=1, alpha=opacity)
		ax2.tick_params(axis="x", labelsize=9)

		# Save the figure
		plt.tight_layout()
		plt.grid(True)
		plt.savefig(to_file, bbox_inches="tight")
		plt.close(fig)

	def summary(self):
		self.model.summary()

	def load(self, filepath):
		self.model.load(filepath)

	def save(self, filepath):
		self.model.save(filepath)

	def load_weights(self, filepath):
		self.model.load_weights(filepath)

	def save_weights(self, filepath):
		self.model.save_weights(filepath)

	def __del__(self):
		del self.map_conv2microconv
		del self.model