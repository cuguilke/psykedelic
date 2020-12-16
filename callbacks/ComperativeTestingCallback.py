"""
Title           :ComperativeTestingCallback.py
Description     :Custom callback to perform performance test for multiple kernel pruning criterias
Author          :Ilke Cugu
Date Created    :24-10-2019
Date Modified   :11-05-2020
version         :1.5.6
python_version  :3.6.6
"""

import keras
import numpy as np
from tools import log, one_hot_to_int
from layers.MicroConv2D import MicroConv2D
from sklearn.metrics import confusion_matrix

class ComperativeTestingCallback(keras.callbacks.Callback):
	"""
	This callback simulates multiple compression scenarios on a single model
	to compare test performances of different pruning criterias.

	Note: Running this callback does not change the properties of the original model.

	# Arguments
		:param tester: (Tester) selected tester for the selected dataset
		:param modes_to_compare: (list) list of selected compression modes
		:param experiment_recorder: (ExperimentRecorder) for cumulative logging of empirical results
	"""
	def __init__(self, tester, modes_to_compare=None, experiment_recorder=None):
		super(ComperativeTestingCallback, self).__init__()
		self.modes_to_compare = modes_to_compare
		self.experiment_recorder = experiment_recorder
		self.tester = tester

	def rename_model(self, name, new_mode):
		new_name = name

		# Replace the mode
		i = new_name.find("[mode=")
		j = new_name[i:].find("]")
		if i != -1 and j != -1:
			active_mode = new_name[i+6:i+j]
			new_name = new_name.replace(active_mode, new_mode)

		return new_name

	def count_params(self):
		count = 0
		for layer in self.model.layers:
			count += layer.count_params()
		return count

	def on_train_end(self, logs=None):
		if self.modes_to_compare is not None:
			vals = {}
			y_test = self.tester.get_y_test()
			y_test = one_hot_to_int(y_test)

			# Save the original model properties before proceeding
			original_param_count = self.count_params()
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
			cm = confusion_matrix(y_test, y_preds)
			denominator = np.sum(cm, axis=1)
			recall = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None
			denominator = np.sum(cm, axis=0)
			precision = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None

			# Baseline performance without compression
			model_name = self.rename_model(self.model.name, "None")
			log("%s Total param: %s" % (model_name, original_param_count))
			log("%s Test loss: %s" % (model_name, original_score[0]))
			log("%s Test accuracy: %s" % (model_name, original_score[1]))
			log("%s Test precision: %s" % (model_name, precision))
			log("%s Test recall: %s" % (model_name, recall))
			vals["None"] = {"param_count": original_param_count,
							"score": 0.0,
							"accuracy": original_score[1],
							"precision": precision,
							"recall": recall}

			# Simulate kernel pruning for each given compression mode
			compression_modes = self.modes_to_compare + ["control_mode"]
			for compression_mode in compression_modes:

				# Trigger dynamic model compression
				layer_names = []
				total_params = []
				active_params = []
				for layer in self.model.layers:
					if isinstance(layer, MicroConv2D):
						layer.set_compression_mode(compression_mode)
						layer.kill_insignificants()

						# Store pruning per layer data
						layer_names.append(layer.name)
						active_params.append(layer.count_params())
						total_params.append(layer.get_total_param_count())

				# Evaluate the hypothetical model with the test data
				score = self.tester.evaluate(self.model)
				param_count = self.count_params()
				compression_score = (score[1] / original_score[1]) * (1.0 - (param_count / original_param_count))

				# Precision - recall scores
				y_preds = self.tester.predict(self.model)
				y_preds = one_hot_to_int(y_preds)
				cm = confusion_matrix(y_test, y_preds)
				denominator = np.sum(cm, axis=1)
				recall = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None
				denominator = np.sum(cm, axis=0)
				precision = np.mean(np.diag(cm) / denominator) if 0 not in denominator else None

				model_name = self.rename_model(self.model.name, compression_mode)
				log("%s Total param: %s" % (model_name, param_count))
				log("%s Test loss: %s" % (model_name, score[0]))
				log("%s Test score: %.4f" % (model_name, compression_score))
				log("%s Test accuracy: %s" % (model_name, score[1]))
				log("%s Test precision: %s" % (model_name, precision))
				log("%s Test recall: %s" % (model_name, recall))

				if compression_mode != "control_mode":
					vals[compression_mode] = {"param_count": param_count,
											  "score": compression_score,
											  "accuracy": score[1],
											  "precision": precision,
											  "recall": recall}

				# Restore the original properties
				self.model.set_weights(original_weights)
				for layer in self.model.layers:
					if isinstance(layer, MicroConv2D):
						layer.set_compression_mode(original_compression_modes[layer.name])
						layer.set_dead_kernels(original_dead_kernels[layer.name])

				# Log the values (when compression_mode is a key in JSON file)
				if self.experiment_recorder is not None and compression_mode != "control_mode":
					# Pruning per layer data
					temp_vals = {"layer_names": layer_names, "active_params": active_params, "total_params": total_params}
					self.experiment_recorder.record(temp_vals, mode="pruning_per_layer", compression_mode=compression_mode)

			# Log the values
			if self.experiment_recorder is not None:
				# Performance data
				self.experiment_recorder.record({"performance": vals}, mode="performance")