"""
Title           :MicroConv2D.py
Description     :Custom conv layer for dynamic model compression
Author          :Ilke Cugu
Date Created    :19-02-2019
Date Modified   :15-05-2020
version         :3.2.3
python_version  :3.6.6
"""

import numpy as np
from keras import backend as K
from keras.layers import Conv2D

class MicroConv2D(Conv2D):
	"""
		New Conv2D implementation with dynamic model compression support
	"""
	def __init__(self, filters, kernel_size,
				 pretrained_weights=None,
				 disable_compression=False,
				 significance_threshold=1e-4,
				 contribution_threshold=1e-2,
				 compression_mode="spectral_norm",
				 compression_rate=0.2,
				 **kwargs):
		"""
		New conv layers where filters die out w.r.t their significance

		# Arguments
			:param filters: (int) the dimensionality of the output space
				(i.e. the number of output filters in the convolution).
			:param kernel_size: (int or tuple/list of 2 ints) specifies the height and width of the 2D convolution window.
            	Can be a single integer to specify the same value for all spatial dimensions.
			:param pretrained_weights: (list of numpy arrays) to deep copy your standard conv layers
			:param disable_compression: (bool) to transform this microconv layer to a standard conv layer
			:param significance_threshold: (float) compression threshold for estimating the kernel's significance
			:param contribution_threshold: (float) compression threshold for kernel contribution to the neuron output
			:param compression_mode: (string) defines the kernel significance w.r.t:
				- "det": abs determinant of the kernel
				- "det_corr": abs determinant of the correlation matrix of the given kernel K (K.T * K)
				- "det_contrib": relative abs determinant of the kernel w.r.t all kernels within a given neuron
				- "det_sorted_kernels": for each neuron bottom X%% of the kernels are killed w.r.t abs determinants
				- "det_sorted_neurons": sum of kernel abs determinants determines the significance and bottom X%% of the neurons are killed
				- "min_eig": min abs eigenvalue of the kernel
				- "min_eig_real": min abs eigenvalue (real parts only) of the kernel
				- "min_eig_contrib": relative min abs eigenvalue of the kernel w.r.t all kernels within a given neuron
				- "min_eig_real_contrib": relative min abs eigenvalue (real parts only) of the kernel w.r.t all kernels within a given neuron
				- "min_eig_sorted_kernels": for each neuron bottom X%% of the kernels are killed w.r.t abs determinants
				- "min_eig_sorted_neurons": sum of kernel min abs eigenvalues determines the significance and bottom X%% of the neurons are killed
				- "spectral_radius": max abs eigenvalue of the kernel
				- "spectral_radius_real": max abs eigenvalue (real parts only) of the kernel
				- "spectral_radius_contrib": relative spectral radius of the kernel w.r.t all kernels within a given neuron
				- "spectral_radius_real_contrib": relative spectral radius (real parts only) of the kernel w.r.t all kernels within a given neuron
				- "spectral_radius_sorted_kernels": for each neuron bottom X%% of the kernels are killed w.r.t spectral radii
				- "spectral_radius_sorted_neurons": sum of kernel spectral radii determines the significance and bottom X%% of the neurons are killed
				- "spectral_norm": max singular value of the kernel
				- "spectral_norm_contrib": relative spectral norm of the kernel w.r.t all kernels within a given neuron
				- "spectral_norm_sorted_kernels": for each neuron bottom X%% of the kernels are killed w.r.t spectral norms
				- "spectral_norm_sorted_neurons": sum of kernel spectral norms determines the significance and bottom X%% of the neurons are killed
				- "weight": sum of abs weights of the kernel
				- "weight_contrib": relative sum of abs weights of the kernel w.r.t all kernels within a given neuron
				- "weight_sorted_kernels": for each neuron bottom X%% of the kernels are killed w.r.t sum of abs kernel weights
				- "weight_sorted_neurons": (Li et al. ICLR 2017) sum of abs kernel weights determines the significance and bottom X%% of the neurons are killed
				- "random_kernels": randomly killing kernels
				- "random_neurons": randomly killing neurons
			:param compression_rate: (float) determines the percentage of the kernels to be killed which is only relevant for:
				- "det_sorted_kernels",
				- "det_sorted_neurons",
				- "min_eig_sorted_kernels",
				- "min_eig_sorted_neurons",
				- "spectral_radius_sorted_kernels",
				- "spectral_radius_sorted_neurons",
				- "spectral_norm_sorted_neurons",
				- "spectral_norm_sorted_kernels",
				- "weight_sorted_kernels",
				- "weight_sorted_neurons",
				- "random_kernels",
				- "random_neurons" compression modes
			:param kwargs: you know what this is
		"""
		super(MicroConv2D, self).__init__(filters, kernel_size, **kwargs)

		# Use pre-trained weights if applicable
		self.pretrained_weights = pretrained_weights
		self.disable_compression = K.variable(1) if disable_compression else K.variable(0)

		# Neural activity map
		self.significance_threshold = significance_threshold
		self.contribution_threshold = contribution_threshold
		self.compression_mode = compression_mode
		self.compression_rate = compression_rate
		self.neuron_count = filters
		self.filter_depth = 0
		self.filter_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
		self.is_alive_kernels = None

		# Bug fix variables
		self.control_counter = 0

	def build(self, input_shape):
		super(MicroConv2D, self).build(input_shape)

		self.filter_depth = input_shape[3]
		self.init_kernel_population_census(self.neuron_count, self.filter_depth)

		if self.pretrained_weights is not None:
			self.set_weights(self.pretrained_weights)

	def init_kernel_population_census(self, neuron_count=None, filter_depth=None):
		"""
		Sets a numpy array to store dead/alive kernel information
		:return:
		"""
		neuron_count = self.neuron_count if neuron_count is None else neuron_count
		filter_depth = self.filter_depth if filter_depth is None else filter_depth
		self.is_alive_kernels = np.ones((neuron_count, filter_depth))

	def get_threshold(self):
		threshold = self.contribution_threshold if "contrib" in self.compression_mode else self.significance_threshold
		return threshold

	def set_threshold(self, threshold):
		if "contrib" in self.compression_mode:
			self.contribution_threshold = threshold
		else:
			self.significance_threshold = threshold

	def get_compression_mode(self):
		return self.compression_mode

	def set_compression_mode(self, compression_mode):
		self.compression_mode = compression_mode

	def set_disable_compression(self, disable=False):
		bool_to_int = 1 if disable else 0
		K.set_value(self.disable_compression , bool_to_int)

	def count_params(self):
		"""
		Counts the number of active parameters in the layer
		:return: int
		"""
		return self.get_total_param_count() - self.get_dead_param_count()

	def get_dead_param_count(self):
		"""
		Counts the parameters in kernels where the kernel is 0 matrix
		:return: int
		"""
		counter = 0
		for i in range(self.neuron_count):
			for j in range(self.filter_depth):
				if self.is_alive_kernels[i][j] == 0:
					counter += 1
		return counter * self.filter_size[0] * self.filter_size[1]

	def get_total_param_count(self):
		return super(MicroConv2D, self).count_params()

	def count_neurons(self):
		"""
		Counts the number of active neurons in the layer
		:return: int
		"""
		return self.get_total_neuron_count() - self.get_dead_neuron_count()

	def get_dead_neuron_count(self):
		"""
		Counts the neurons where all kernels are 0 matrices
		:return: int
		"""
		counter = 0
		dead_neuron = np.zeros(self.filter_depth)
		for i in range(self.neuron_count):
			if np.array_equal(self.is_alive_kernels[i], dead_neuron):
				counter += 1
		return counter

	def get_total_neuron_count(self):
		"""
		Returns the number of total neurons (active + dead) in the layer
		:return: int
		"""
		return self.neuron_count

	def is_significant(self, kernel, with_respect_to=None):
		"""
		Decides if the given kernel is significant
		:param kernel: 2D numpy array  (only a single kernel is given)
		:param with_respect_to: (optional) for relative significance computation
		:return: bool
		"""
		result = True

		try:
			if self.compression_mode == "det":
				n = kernel.shape[0]
				determinant = np.linalg.det(kernel)
				result = np.absolute(determinant) >= pow(self.significance_threshold, n)

			elif self.compression_mode == "det_corr":
				n = kernel.shape[0]
				determinant = np.linalg.det(np.dot(kernel.T, kernel))
				result = np.absolute(determinant) >= pow(self.significance_threshold, 2*n)

			elif self.compression_mode == "det_contrib":
				determinant = np.linalg.det(kernel)
				result = (np.absolute(determinant) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "min_eig":
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result = np.min(eigenvalues) >= self.significance_threshold

			elif self.compression_mode == "min_eig_contrib":
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result = (np.min(eigenvalues) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "min_eig_real":
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result = np.min(eigenvalues) >= self.significance_threshold

			elif self.compression_mode == "min_eig_real_contrib":
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result = (np.min(eigenvalues) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "spectral_radius":
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result = np.max(eigenvalues) >= self.significance_threshold

			elif self.compression_mode == "spectral_radius_contrib":
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result = (np.max(eigenvalues) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "spectral_radius_real":
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result = np.max(eigenvalues) >= self.significance_threshold

			elif self.compression_mode == "spectral_radius_real_contrib":
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result = (np.max(eigenvalues) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "spectral_norm":
				spectral_norm = np.linalg.norm(kernel, 2)
				result = spectral_norm >= self.significance_threshold

			elif self.compression_mode == "spectral_norm_contrib":
				spectral_norm = np.linalg.norm(kernel, 2)
				result = (spectral_norm / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "weight":
				weights = np.absolute(kernel)
				result = np.average(weights) >= self.significance_threshold

			elif self.compression_mode == "weight_contrib":
				weights = np.absolute(kernel)
				result = (np.average(weights) / with_respect_to) >= self.contribution_threshold

			elif self.compression_mode == "control_mode":
				# There is no static implementation for this mode
				# Instead, you this option to bug fix

				# ------------------------------------------------------------- #
				# Check compression mode: det
				"""
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				eigenvalues = [0 if x == 0 else np.log10(x) for x in eigenvalues]
				result = np.sum(eigenvalues) >= np.log10(self.significance_threshold)
				"""
				# ------------------------------------------------------------- #

				# ------------------------------------------------------------- #
				# Check compression mode: Harris corner detection
				"""
				determinant = np.linalg.det(kernel)
				trace = np.trace(kernel)
				result = np.absolute(determinant - 0.027 * np.power(trace, 3)) >= self.significance_threshold
				"""
				# ------------------------------------------------------------- #

				# ------------------------------------------------------------- #
				# Check compression mode: trace
				trace = np.trace(kernel)
				result = np.absolute(trace) >= self.significance_threshold

		# ------------------------------------------------------------- #

		except Exception as e:
			result = True
			print(e)
			print("-----------------------------------------")
			print("Function args:")
			print("=============")
			print("kernel:", kernel)
			print("with_respect_to", with_respect_to)

		return result

	def get_total_det(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				determinant = np.linalg.det(kernel)
				result += np.absolute(determinant)
		return result

	def get_total_det_corr(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				determinant = np.linalg.det(np.dot(kernel.T, kernel))
				result += np.absolute(determinant)
		return result

	def get_total_min_eig(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result += np.min(eigenvalues)
		return result

	def get_total_min_eig_real(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result += np.min(eigenvalues)
		return result

	def get_total_spectral_radius(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				eigenvalues = np.absolute(np.linalg.eigvals(kernel))
				result += np.max(eigenvalues)
		return result

	def get_total_spectral_radius_real(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				eigenvalues = np.absolute(np.real(np.linalg.eigvals(kernel)))
				result += np.max(eigenvalues)
		return result

	def get_total_spectral_norm(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = kernels[:, :, i]
				result += np.linalg.norm(kernel, 2)
		return result

	def get_total_weight(self, kernels, n, neuron_id):
		result = 0
		for i in range(n):
			if self.is_alive_kernels[neuron_id][i] == 1:
				kernel = np.absolute(kernels[:, :, i])
				result += np.sum(kernel)
		return result

	def get_total(self, kernels, n, neuron_id):
		result = 1

		if "det_corr" in self.compression_mode:
			result = self.get_total_det_corr(kernels, n, neuron_id)
		elif "det" in self.compression_mode:
			result = self.get_total_det(kernels, n, neuron_id)
		elif "min_eig_real" in self.compression_mode:
			result = self.get_total_min_eig_real(kernels, n, neuron_id)
		elif "min_eig" in self.compression_mode:
			result = self.get_total_min_eig(kernels, n, neuron_id)
		elif "spectral_radius_real" in self.compression_mode:
			result = self.get_total_spectral_radius_real(kernels, n, neuron_id)
		elif "spectral_radius" in self.compression_mode:
			result = self.get_total_spectral_radius(kernels, n, neuron_id)
		elif "spectral_norm" in self.compression_mode:
			result = self.get_total_spectral_norm(kernels, n, neuron_id)
		elif "weight" in self.compression_mode:
			result = self.get_total_weight(kernels, n, neuron_id)

		return result

	def sort_kernels(self, kernels, n, neuron_id):
		"""
		Sorts the kernels w.r.t. the given criteria
		:param kernels: Numpy array of kernels in a single neuron
		:param n: # of kernels in a given neuron
		:param neuron_id: index of the given neuron
		:return: List that contains kernel indices sorted w.r.t. the given criteria
		"""
		sorted_kernel_indices = []
		vals = []

		if self.compression_mode == "det_sorted_kernels":
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					kernel = kernels[:, :, i]
					vals.append(np.absolute(np.linalg.det(kernel)))
					sorted_kernel_indices.append(i)
			sorted_kernel_indices = [x for _, x in sorted(zip(vals, sorted_kernel_indices), key=lambda pair: pair[0])]

		elif self.compression_mode == "min_eig_sorted_kernels":
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					kernel = kernels[:, :, i]
					eigenvalues = np.absolute(np.linalg.eigvals(kernel))
					vals.append(np.min(eigenvalues))
					sorted_kernel_indices.append(i)
			sorted_kernel_indices = [x for _, x in sorted(zip(vals, sorted_kernel_indices), key=lambda pair: pair[0])]

		elif self.compression_mode == "spectral_radius_sorted_kernels":
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					kernel = kernels[:, :, i]
					eigenvalues = np.absolute(np.linalg.eigvals(kernel))
					vals.append(np.max(eigenvalues))
					sorted_kernel_indices.append(i)
			sorted_kernel_indices = [x for _, x in sorted(zip(vals, sorted_kernel_indices), key=lambda pair: pair[0])]

		elif self.compression_mode == "spectral_norm_sorted_kernels":
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					kernel = kernels[:, :, i]
					spectral_norm = np.linalg.norm(kernel, 2)
					vals.append(spectral_norm)
					sorted_kernel_indices.append(i)
			sorted_kernel_indices = [x for _, x in sorted(zip(vals, sorted_kernel_indices), key=lambda pair: pair[0])]

		elif self.compression_mode == "weight_sorted_kernels":
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					kernel = np.absolute(kernels[:, :, i])
					vals.append(np.average(kernel))
					sorted_kernel_indices.append(i)
			sorted_kernel_indices = [x for _, x in sorted(zip(vals, sorted_kernel_indices), key=lambda pair: pair[0])]

		else: # self.compression_mode == "random_kernels"
			for i in range(n):
				if self.is_alive_kernels[neuron_id][i] == 1:
					sorted_kernel_indices.append(i)

		return sorted_kernel_indices

	def get_dead_kernels(self):
		return self.is_alive_kernels

	def set_dead_kernels(self, is_alive_kernels):
		self.is_alive_kernels = np.copy(is_alive_kernels)

	def mask_dead_kernels(self):
		current_weights = self.get_weights()
		kernel_shape = current_weights[0].shape

		for neuron in range(kernel_shape[3]):
			for depth in range(kernel_shape[2]):
				if self.is_alive_kernels[neuron][depth] == 0:
					current_weights[0][:, :, depth, neuron] = np.zeros((kernel_shape[0], kernel_shape[1]))

		self.set_weights(current_weights)

	def kill_insignificants(self):
		current_weights = self.get_weights()
		kernel_shape = current_weights[0].shape

		if self.compression_mode in ["det", "det_corr", "min_eig", "min_eig_real", "spectral_radius", "spectral_radius_real", "spectral_norm", "weight", "control_mode"]:
			for neuron in range(kernel_shape[3]):
				for depth in range(kernel_shape[2]):
					# Get the 3x3 convolution matrix
					kernel = current_weights[0][:, :, depth, neuron]

					# Kill the kernel if insignificant
					if self.is_alive_kernels[neuron][depth] == 1:
						if not self.is_significant(kernel):
							self.is_alive_kernels[neuron][depth] = 0
							current_weights[0][:, :, depth, neuron] = np.zeros((kernel.shape[0], kernel.shape[1]))

		elif self.compression_mode in ["det_contrib", "min_eig_contrib", "min_eig_real_contrib", "spectral_radius_contrib", "spectral_radius_real_contrib", "spectral_norm_contrib", "weight_contrib"]:
			for neuron in range(kernel_shape[3]):
				# Get kernels for all channels
				kernels = current_weights[0][:, :, :, neuron]

				# Compute the denominator for relative significance
				with_respect_to = self.get_total(kernels, kernel_shape[2], neuron)

				for depth in range(kernel_shape[2]):
					# Get the 3x3 convolution matrix
					kernel = current_weights[0][:, :, depth, neuron]

					# Kill the kernel if insignificant
					if self.is_alive_kernels[neuron][depth] == 1:
						if not self.is_significant(kernel, with_respect_to):
							self.is_alive_kernels[neuron][depth] = 0
							current_weights[0][:, :, depth, neuron] = np.zeros((kernel.shape[0], kernel.shape[1]))

		elif self.compression_mode in ["det_sorted_kernels", "min_eig_sorted_kernels", "spectral_radius_sorted_kernels", "spectral_norm_sorted_kernels", "weight_sorted_kernels", "random_kernels"]:
			for neuron in range(kernel_shape[3]):
				# Get kernels for all channels
				kernels = current_weights[0][:, :, :, neuron]

				# Get the sorted kernel indices and determine the bottom 10%
				sorted_kernel_indices = self.sort_kernels(kernels, kernel_shape[2], neuron)
				kernels_to_be_killed = sorted_kernel_indices[0:int(len(sorted_kernel_indices) * (1.0 - self.compression_rate))]

				for depth in kernels_to_be_killed:
					# Get the 3x3 convolution matrix
					kernel = current_weights[0][:, :, depth, neuron]

					# Kill the kernel
					self.is_alive_kernels[neuron][depth] = 0
					current_weights[0][:, :, depth, neuron] = np.zeros((kernel.shape[0], kernel.shape[1]))

		elif self.compression_mode in ["det_sorted_neurons", "min_eig_sorted_neurons", "spectral_radius_sorted_neurons", "spectral_norm_sorted_neurons", "weight_sorted_neurons", "random_neurons"]:
			# TODO: Implement this
			for neuron in range(kernel_shape[3]):
				# Get sum of the selected criteria for the neuron at hand
				kernels = current_weights[0][:, :, :, neuron]
				neural_value = self.get_total(kernels, kernel_shape[2],neuron)

		self.set_weights(current_weights)

	def neural_activity_check(self):
		current_weights = self.get_weights()
		kernel_shape = current_weights[0].shape
		result = True

		for neuron in range(kernel_shape[3]):
			for depth in range(kernel_shape[2]):
				kernel = current_weights[0][:, :, depth, neuron]
				if self.is_alive_kernels[neuron][depth] == 0:
					result = result and np.array_equal(current_weights[0][:, :, depth, neuron], np.zeros((kernel.shape[0], kernel.shape[1])))

		return result

	def report_compression_stats(self):
		report = ">> Layer: %s, active neurons: %s/%s, active parameters: %s/%s" % (self.name, self.count_neurons(), self.get_total_neuron_count(), self.count_params(), self.get_total_param_count())
		return report

	def call(self, inputs):
		self.mask_dead_kernels()

		return super(MicroConv2D, self).call(inputs)