"""
Title           :ExperimentRecorder.py
Description     :JSON exporter class to store experimental results over time
Author          :Ilke Cugu
Date Created    :15-01-2020
Date Modified   :16-06-2020
version         :1.1.4
python_version  :3.6.6
"""

import os
import json

class ExperimentRecorder:
	"""
	Custom JSON exporter class to store information on MicroResNet experiments

	# Arguments
		:param model_name: (string) Keras model name
		:param optimizer: (string) name of the selected Keras optimizer
		:param loss: (string) name of the selected Keras loss function
		:param base_lr: (float) learning rate
		:param batch_size: (int) # of inputs in a mini-batch
		:param epochs: (int) # of full training passes
		:param init_mode: (string) {random_init | static_init}
		:param dataset: (string) selected dataset
		:param l1_penalty: (float) regularization coefficient
		:param history_interval: (int) for history callbacks, defines the interval of logs
		:param threshold: (float) {significance_threshold | contribution_threshold}
		:param compression_mode: (string) selected compression mode
		:param verbose: (int) Keras verbose argument
		:param path: (string) absolute path to 'experiment.json' file if already exists
	"""
	def __init__(self,
				 model_name,
				 optimizer,
				 loss,
				 base_lr,
				 batch_size,
				 epochs,
				 init_mode,
				 dataset,
				 l1_penalty,
				 history_interval=1,
				 threshold=None,
				 compression_mode=None,
				 verbose=0,
				 path="experiment.json"):

		self.model_name = model_name
		self.optimizer = optimizer
		self.loss = loss
		self.base_lr = "{:.0e}".format(base_lr)
		self.batch_size = batch_size
		self.epochs = str(epochs)
		self.init_mode = init_mode
		self.dataset = dataset
		self.history_interval = str(history_interval)
		self.l1_penalty = "{:.0e}".format(l1_penalty)
		self.threshold = "{:.0e}".format(threshold)
		self.compression_mode = compression_mode
		self.verbose = verbose
		self.path = path
		self.hist_cache = {}

		if os.path.isfile(self.path):
			self.load_data()

		self.model = "%s:%s:%s:%s" % (self.model_name, self.optimizer, self.loss, self.base_lr)
		self.base_key = ":".join([self.model, str(self.epochs), str(self.batch_size), self.init_mode, self.dataset, str(self.l1_penalty)])

	def load_data(self):
		"""
		Loads the current experiment history to append new results
		"""
		with open(self.path, "r") as hist_file:
			self.hist_cache = json.load(hist_file)

	def save_data(self):
		"""
		Saves the accumulated data by updating the existing file located at self.path
		"""
		with open(self.path, "w+") as hist_file:
			json.dump(self.hist_cache, hist_file)

	def record(self, data, mode, compression_mode=None):
		if mode == "learning_curve":
			self.record_learning_curve(data["hist"])
		elif mode == "pruning_per_layer":
			self.record_pruning_per_layer(data["layer_names"], data["active_params"], data["total_params"], compression_mode)
		elif mode == "pruning_per_layer_history":
			self.record_pruning_per_layer_history(data["pruning_per_layer_history"])
		elif mode == "eig_analysis":
			self.record_eig_analysis(data["layer_names"], data["active_real"], data["pruned_real"], data["active_complex"], data["pruned_complex"])
		elif mode == "eig_stats":
			self.record_eig_stats(data["layer_names"], data["total_complex_list"], data["total_real_list"], data["target_complex_list"], data["pruned_complex_list"], data["target_real_list"], data["pruned_real_list"])
		elif mode == "set_analysis":
			self.record_set_analysis(data["group_sizes"], data["groups"], data["codes"])
		elif mode == "performance":
			self.record_performance(data["performance"])
		elif mode == "performance_history":
			self.record_performance_history(data["performance_history"])
		elif mode == "pruning_per_threshold":
			self.record_pruning_per_threshold(data["final_vals"], data["thresholds"])

	def record_learning_curve(self, hist):
		if "learning_curve" in self.hist_cache:
			if self.base_key in self.hist_cache["learning_curve"]:
				if self.threshold in self.hist_cache["learning_curve"][self.base_key]:
					if self.compression_mode in self.hist_cache["learning_curve"][self.base_key][self.threshold]:
						self.hist_cache["learning_curve"][self.base_key][self.threshold][self.compression_mode].append(hist)
					else:
						self.hist_cache["learning_curve"][self.base_key][self.threshold][self.compression_mode] = [hist]
				else:
					self.hist_cache["learning_curve"][self.base_key][self.threshold] = {self.compression_mode: [hist]}
			else:
				self.hist_cache["learning_curve"][self.base_key] = {self.threshold: {self.compression_mode: [hist]}}
		else:
			self.hist_cache["learning_curve"] ={self.base_key: {self.threshold: {self.compression_mode: [hist]}}}

	def record_pruning_per_layer(self, layer_names, active_params, total_params, compression_mode=None):
		"""
		# Arguments
			:param layer_names:
			:param active_params:
			:param total_params:
			:param compression_mode: (string) Optional. ComperativeTestingCallback set this to record data for multiple modes
		"""
		compression_mode = self.compression_mode if compression_mode is None else compression_mode
		entry = {"layer_names": layer_names, "active_params": active_params, "total_params": total_params}

		if "pruning_per_layer" in self.hist_cache:
			if self.base_key in self.hist_cache["pruning_per_layer"]:
				if self.threshold in self.hist_cache["pruning_per_layer"][self.base_key]:
					if compression_mode in self.hist_cache["pruning_per_layer"][self.base_key][self.threshold]:
						self.hist_cache["pruning_per_layer"][self.base_key][self.threshold][compression_mode].append(entry)
					else:
						self.hist_cache["pruning_per_layer"][self.base_key][self.threshold][compression_mode] = [entry]
				else:
					self.hist_cache["pruning_per_layer"][self.base_key][self.threshold] = {compression_mode: [entry]}
			else:
				self.hist_cache["pruning_per_layer"][self.base_key] = {self.threshold: {compression_mode: [entry]}}
		else:
			self.hist_cache["pruning_per_layer"] ={self.base_key: {self.threshold: {compression_mode: [entry]}}}

	def record_pruning_per_layer_history(self, pruning_per_layer_history):
		entry = pruning_per_layer_history

		if "pruning_per_layer_history" in self.hist_cache:
			if self.base_key in self.hist_cache["pruning_per_layer_history"]:
				if self.threshold in self.hist_cache["pruning_per_layer_history"][self.base_key]:
					if self.history_interval in self.hist_cache["pruning_per_layer_history"][self.base_key][self.threshold]:
						self.hist_cache["pruning_per_layer_history"][self.base_key][self.threshold][self.history_interval].append(entry)
					else:
						self.hist_cache["pruning_per_layer_history"][self.base_key][self.threshold][self.history_interval] = [entry]
				else:
					self.hist_cache["pruning_per_layer_history"][self.base_key][self.threshold] = {self.history_interval: [entry]}
			else:
				self.hist_cache["pruning_per_layer_history"][self.base_key] = {self.threshold: {self.history_interval: [entry]}}
		else:
			self.hist_cache["pruning_per_layer_history"] ={self.base_key: {self.threshold: {self.history_interval: [entry]}}}

	def record_eig_analysis(self, layer_names, active_real, pruned_real, active_complex, pruned_complex):
		entry = {"layer_names": layer_names, "active_real": active_real, "pruned_real": pruned_real, "active_complex": active_complex, "pruned_complex": pruned_complex}

		if "eig_analysis" in self.hist_cache:
			if self.base_key in self.hist_cache["eig_analysis"]:
				if self.threshold in self.hist_cache["eig_analysis"][self.base_key]:
					if self.compression_mode in self.hist_cache["eig_analysis"][self.base_key][self.threshold]:
						self.hist_cache["eig_analysis"][self.base_key][self.threshold][self.compression_mode].append(entry)
					else:
						self.hist_cache["eig_analysis"][self.base_key][self.threshold][self.compression_mode] = [entry]
				else:
					self.hist_cache["eig_analysis"][self.base_key][self.threshold] = {self.compression_mode: [entry]}
			else:
				self.hist_cache["eig_analysis"][self.base_key] = {self.threshold: {self.compression_mode: [entry]}}
		else:
			self.hist_cache["eig_analysis"] ={self.base_key: {self.threshold: {self.compression_mode: [entry]}}}

	def record_eig_stats(self, layer_names, total_complex_list, total_real_list, target_complex_list, pruned_complex_list, target_real_list, pruned_real_list):
		entry = {
			"layer_names": layer_names,
			"total_complex_list": total_complex_list,
			"total_real_list": total_real_list,
			"target_complex_list": target_complex_list,
			"pruned_complex_list": pruned_complex_list,
			"target_real_list": target_real_list,
			"pruned_real_list": pruned_real_list
		}

		if "eig_stats" in self.hist_cache:
			if self.base_key in self.hist_cache["eig_stats"]:
				if self.threshold in self.hist_cache["eig_stats"][self.base_key]:
					self.hist_cache["eig_stats"][self.base_key][self.threshold].append(entry)
				else:
					self.hist_cache["eig_stats"][self.base_key][self.threshold] = [entry]
			else:
				self.hist_cache["eig_stats"][self.base_key] = {self.threshold: [entry]}
		else:
			self.hist_cache["eig_stats"] = {self.base_key: {self.threshold: [entry]}}

	def record_set_analysis(self, group_sizes, groups, codes):
		entry = {"group_sizes": group_sizes, "groups": groups, "codes": codes}

		if "set_analysis" in self.hist_cache:
			if self.base_key in self.hist_cache["set_analysis"]:
				if self.threshold in self.hist_cache["set_analysis"][self.base_key]:
					self.hist_cache["set_analysis"][self.base_key][self.threshold].append(entry)
				else:
					self.hist_cache["set_analysis"][self.base_key][self.threshold] = [entry]
			else:
				self.hist_cache["set_analysis"][self.base_key] = {self.threshold: [entry]}
		else:
			self.hist_cache["set_analysis"] = {self.base_key: {self.threshold: [entry]}}

	def record_performance(self, performance):
		entry = performance

		if "performance" in self.hist_cache:
			if self.base_key in self.hist_cache["performance"]:
				if self.threshold in self.hist_cache["performance"][self.base_key]:
					self.hist_cache["performance"][self.base_key][self.threshold].append(entry)
				else:
					self.hist_cache["performance"][self.base_key][self.threshold] = [entry]
			else:
				self.hist_cache["performance"][self.base_key] = {self.threshold: [entry]}
		else:
			self.hist_cache["performance"] = {self.base_key: {self.threshold: [entry]}}

	def record_performance_history(self, performance_history):
		entry = performance_history

		if "performance_history" in self.hist_cache:
			if self.base_key in self.hist_cache["performance_history"]:
				if self.threshold in self.hist_cache["performance_history"][self.base_key]:
					if self.history_interval in self.hist_cache["performance_history"][self.base_key][self.threshold]:
						self.hist_cache["performance_history"][self.base_key][self.threshold][self.history_interval].append(entry)
					else:
						self.hist_cache["performance_history"][self.base_key][self.threshold][self.history_interval] = entry
				else:
					self.hist_cache["performance_history"][self.base_key][self.threshold] = {self.history_interval: [entry]}
			else:
				self.hist_cache["performance_history"][self.base_key] = {self.threshold: {self.history_interval: [entry]}}
		else:
			self.hist_cache["performance_history"] = {self.base_key: {self.threshold: {self.history_interval: [entry]}}}

	def record_pruning_per_threshold(self, final_vals, thresholds):
		entry = {"final_vals": final_vals, "thresholds": thresholds}

		if "pruning_per_threshold" in self.hist_cache:
			if self.base_key in self.hist_cache["pruning_per_threshold"]:
				self.hist_cache["pruning_per_threshold"][self.base_key].append(entry)
			else:
				self.hist_cache["pruning_per_threshold"][self.base_key] = [entry]
		else:
			self.hist_cache["pruning_per_threshold"] = {self.base_key: [entry]}