"""
Title           :HistoryCallback.py
Description     :Custom callback to observe the evolution of pruning through epochs
Author          :Ilke Cugu
Date Created    :11-05-2020
Date Modified   :15-06-2020
version         :1.0.3
python_version  :3.6.6
"""

import numpy as np
from tools import one_hot_to_int
from overrides import overrides
from layers.MicroConv2D import MicroConv2D
from sklearn.metrics import confusion_matrix
from callbacks.EigenvalueCallback import EigenvalueCallback

class HistoryCallback(EigenvalueCallback):
	"""
	This callback tracks internal properties of CNNs through epochs

	Note: Running this callback does not change the properties of the original model.

	# Arguments
		:param model_name: (string) model name for the resulting charts
		:param tester: (Tester) selected tester for the selected dataset
		:param epochs: (int) total number of epochs
		:param step_size: (int) epochs // step_size = frequency of the callback
		:param modes_to_compare: (list) list of selected compression modes
		:param to_dir: (string) dir path to save the charts
		:param experiment_recorder: (ExperimentRecorder) for cumulative logging of empirical results
	"""
	def __init__(self, model_name, tester, epochs, step_size=1, modes_to_compare=None, to_dir="", experiment_recorder=None):
		super(HistoryCallback, self).__init__(model_name, to_dir, experiment_recorder)
		self.modes_to_compare = modes_to_compare
		self.tester = tester
		self.epochs = epochs
		self.step_size = step_size
		self.y_test = one_hot_to_int(self.tester.get_y_test())
		self.original_param_count = None
		self.experiment_data = None

	def on_train_begin(self, logs=None):
		self.original_param_count = self.count_params()

		# Data storage
		num_steps = (self.epochs // self.step_size)
		temp = ["None"] + self.modes_to_compare
		self.experiment_data = {
			"performance_history": {
				"param_count": {c: [0.]*num_steps for c in temp},
				"score": {c: [0.]*num_steps for c in temp},
				"accuracy": {c: [0.]*num_steps for c in temp},
				"precision": {c: [0.]*num_steps for c in temp},
				"recall": {c: [0.]*num_steps for c in temp}
			},
			"pruning_per_layer_history": {
				"layer_names": self.get_layer_names(),
				"total_params": self.get_total_params(),
				"active_params": {c: np.zeros((num_steps, self.count_layers())).tolist() for c in self.modes_to_compare}
			}
		}

	def count_params(self):
		count = 0
		for layer in self.model.layers:
			count += layer.count_params()
		return count

	def count_layers(self):
		count = 0
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				count += 1
		return count

	def get_layer_names(self):
		layer_names = []
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				layer_names.append(layer.name)
		return layer_names

	def get_total_params(self):
		total_params = []
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				total_params.append(layer.get_total_param_count())
		return total_params

	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.step_size == 0 and self.modes_to_compare is not None:
			step = epoch // self.step_size

			# Save the original model properties before proceeding
			original_score = self.tester.evaluate(self.model)
			original_weights = np.copy(self.model.get_weights())
			original_compression_modes = {}
			original_dead_kernels = {}
			for layer in self.model.layers:
				if isinstance(layer, MicroConv2D):
					original_compression_modes[layer.name] = layer.get_compression_mode()
					original_dead_kernels[layer.name] = np.copy(layer.get_dead_kernels())

			# Precision - recall scores
			y_preds = self.tester.predict(self.model)
			y_preds = one_hot_to_int(y_preds)
			cm = confusion_matrix(self.y_test, y_preds)
			denominator = np.sum(cm, axis=1)
			recall = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None
			denominator = np.sum(cm, axis=0)
			precision = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None

			# Baseline performance without compression
			self.experiment_data["performance_history"]["param_count"]["None"][step] = self.original_param_count
			self.experiment_data["performance_history"]["score"]["None"][step] = 0.0
			self.experiment_data["performance_history"]["accuracy"]["None"][step] = original_score[1]
			self.experiment_data["performance_history"]["precision"]["None"][step] = recall
			self.experiment_data["performance_history"]["recall"]["None"][step] = precision

			# Simulate kernel pruning for each given compression mode
			for compression_mode in self.modes_to_compare:

				# Trigger dynamic model compression
				active_params = []
				for layer in self.model.layers:
					if isinstance(layer, MicroConv2D):
						layer.set_compression_mode(compression_mode)
						layer.kill_insignificants()

						# Store pruning per layer data
						active_params.append(layer.count_params())

				# Evaluate the hypothetical model with the test data
				score = self.tester.evaluate(self.model)
				param_count = self.count_params()
				compression_score = (score[1] / original_score[1]) * (1.0 - (param_count / self.original_param_count))

				# Precision - recall scores
				y_preds = self.tester.predict(self.model)
				y_preds = one_hot_to_int(y_preds)
				cm = confusion_matrix(self.y_test, y_preds)
				denominator = np.sum(cm, axis=1)
				recall = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None
				denominator = np.sum(cm, axis=0)
				precision = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None

				if compression_mode != "control_mode":
					# Performance history
					self.experiment_data["performance_history"]["param_count"][compression_mode][step] = param_count
					self.experiment_data["performance_history"]["score"][compression_mode][step] = compression_score
					self.experiment_data["performance_history"]["accuracy"][compression_mode][step] = score[1]
					self.experiment_data["performance_history"]["precision"][compression_mode][step] = precision
					self.experiment_data["performance_history"]["recall"][compression_mode][step] = recall

					# Pruning per layer history
					self.experiment_data["pruning_per_layer_history"]["active_params"][compression_mode][step] = active_params

				# Restore the original properties
				self.model.set_weights(original_weights)
				for layer in self.model.layers:
					if isinstance(layer, MicroConv2D):
						layer.set_compression_mode(original_compression_modes[layer.name])
						layer.set_dead_kernels(original_dead_kernels[layer.name])

	@overrides
	def on_train_end(self, logs=None):
		# Log the accumulated values
		if self.experiment_recorder is not None:
			# Performance data
			self.experiment_recorder.record({"performance_history": self.experiment_data["performance_history"]}, mode="performance_history")

			# Pruning per layer data
			self.experiment_recorder.record({"pruning_per_layer_history": self.experiment_data["pruning_per_layer_history"]}, mode="pruning_per_layer_history")