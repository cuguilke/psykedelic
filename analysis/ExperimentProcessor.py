"""
Title           :ExperimentProcessor.py
Description     :Analysis and visualization tool for cumulative empirical results of ExperimentRecorder
Author          :Ilke Cugu
Date Created    :12-02-2020
Date Modified   :09-11-2020
version         :1.3.6
python_version  :3.6.6
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from tools import to_scientific

class ExperimentProcessor:
	"""
	Custom class to analyze and visualiza the cumulative experimental results

	# Arguments
		:param model: (string)
		:param optimizer: (string)
		:param loss: (string)
		:param lr: (float)
		:param epochs: (int)
		:param batch_size: (int)
		:param init_mode: (string)
		:param dataset: (string)
		:param l1_penalty: (float)
		:param threshold: (float)
		:param to_dir: (string)
		:param path: (string) absolute path to 'experiment.json' file if already exists
	"""
	def __init__(self,
				 model=None,
				 optimizer=None,
				 loss=None,
				 lr=None,
				 epochs=None,
				 batch_size=None,
				 init_mode=None,
				 dataset=None,
				 l1_penalty=None,
				 threshold=None,
				 to_dir=".",
				 path="experiment.json"):

		self.experiment_config = {"model": model,
								  "optimizer": optimizer,
								  "loss": loss,
								  "lr": lr,
								  "epochs": epochs,
								  "batch_size": batch_size,
								  "init_mode": init_mode,
								  "dataset": dataset,
								  "l1_penalty": l1_penalty,
								  "threshold": threshold}
		self.to_dir = to_dir
		self.path = path
		self.hist = {}

		# Chart config
		self.tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
						  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
						  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
						  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
						  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
		self.hatches = ["", "++", "--", "//", "\\", "o", "x", "-", "+", ".", "/", "O", "*", ".."]

		# Process colors
		for i in range(len(self.tableau20)):
			r, g, b = self.tableau20[i]
			self.tableau20[i] = (r / 255., g / 255., b / 255.)

		if os.path.isfile(self.path):
			self.load_data()

	def save_data(self):
		"""
		Overwrites the experiment history
		"""
		with open(self.path, "w") as hist_file:
			json.dump(self.hist, hist_file)

	def load_data(self):
		"""
		Loads the current experiment history to append new results
		"""
		with open(self.path, "r") as hist_file:
			self.hist = json.load(hist_file)

	def merge_logs(self, path):
		"""
		Merges separate histories (obtained by different machines)

		# Arguments
			:param path: (string) absolute path to the JSON file you want to merge with
		"""
		print("Merge started...")
		with open(path) as hist_file:
			additional_data = json.load(hist_file)

		for chart_type in additional_data:
			print("Merging %s..." % chart_type)
			if chart_type == "eig_analysis":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:

									for compression_mode in additional_data[chart_type][experiment][threshold]:
										if compression_mode in self.hist[chart_type][experiment][threshold]:
											self.hist[chart_type][experiment][threshold][compression_mode].extend(additional_data[chart_type][experiment][threshold][compression_mode])
										else:
											self.hist[chart_type][experiment][threshold][compression_mode] = additional_data[chart_type][experiment][threshold][compression_mode]
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			if chart_type == "eig_stats":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:
									self.hist[chart_type][experiment][threshold].extend(additional_data[chart_type][experiment][threshold])
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "set_analysis":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:
									self.hist[chart_type][experiment][threshold].extend(additional_data[chart_type][experiment][threshold])
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "pruning_per_threshold":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:
							self.hist[chart_type][experiment].extend(additional_data[chart_type][experiment])
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "pruning_per_layer":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:

									for compression_mode in additional_data[chart_type][experiment][threshold]:
										if compression_mode in self.hist[chart_type][experiment][threshold]:
											self.hist[chart_type][experiment][threshold][compression_mode].extend(additional_data[chart_type][experiment][threshold][compression_mode])
										else:
											self.hist[chart_type][experiment][threshold][compression_mode] = additional_data[chart_type][experiment][threshold][compression_mode]
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "pruning_per_layer_history":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:

									for step_size in additional_data[chart_type][experiment][threshold]:
										if step_size in self.hist[chart_type][experiment][threshold]:
											self.hist[chart_type][experiment][threshold][step_size].extend(additional_data[chart_type][experiment][threshold][step_size])
										else:
											self.hist[chart_type][experiment][threshold][step_size] = additional_data[chart_type][experiment][threshold][step_size]
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "performance":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:
									self.hist[chart_type][experiment][threshold].extend(additional_data[chart_type][experiment][threshold])
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "performance_history":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:

									for step_size in additional_data[chart_type][experiment][threshold]:
										if step_size in self.hist[chart_type][experiment][threshold]:
											self.hist[chart_type][experiment][threshold][step_size].extend(additional_data[chart_type][experiment][threshold][step_size])
										else:
											self.hist[chart_type][experiment][threshold][step_size] = additional_data[chart_type][experiment][threshold][step_size]
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

			elif chart_type == "learning_curve":
				if chart_type in self.hist:

					for experiment in additional_data[chart_type]:
						if experiment in self.hist[chart_type]:

							for threshold in additional_data[chart_type][experiment]:
								if threshold in self.hist[chart_type][experiment]:

									for compression_mode in additional_data[chart_type][experiment][threshold]:
										if compression_mode in self.hist[chart_type][experiment][threshold]:
											self.hist[chart_type][experiment][threshold][compression_mode].extend(additional_data[chart_type][experiment][threshold][compression_mode])
										else:
											self.hist[chart_type][experiment][threshold][compression_mode] = additional_data[chart_type][experiment][threshold][compression_mode]
								else:
									self.hist[chart_type][experiment][threshold] = additional_data[chart_type][experiment][threshold]
						else:
							self.hist[chart_type][experiment] = additional_data[chart_type][experiment]
				else:
					self.hist[chart_type] = additional_data[chart_type]

		# Save the changes
		self.save_data()
		print("Done.")

	def update_logs(self,
					update_learning_curve=False,
					update_pruning_per_layer=False,
					update_eig_analysis=False,
					update_set_analysis=False,
					update_performance=False,
					update_performance_history=False,
					update_pruning_per_threshold=False):
		"""
		Iterates through old recordings and update the structure according to the latest requirements
		"""
		there_is_a_change = False

		if update_learning_curve:
			pass

		if update_pruning_per_layer:
			pass

		if update_eig_analysis:
			pass

		if update_set_analysis:
			if "set_analysis" in self.hist:
				# Update: 22.04.2020 ----------------------------- #
				pre_update_codes = {"a": "min_eig",
									"b": "weight",
									"c": "det",
									"d": "spectral_radius",
									"e": "spectral_norm",
									"f": "det_corr"}

				for experiment in self.hist["set_analysis"]:
					for threshold in self.hist["set_analysis"][experiment]:
						for entry in self.hist["set_analysis"][experiment][threshold]:
							if "codes" not in entry:
								entry["codes"] = pre_update_codes
								there_is_a_change = True
				# ------------------------------------------------ #

		if update_performance:
			pass

		if update_performance_history:
			if "performance_history" in self.hist:
				# Update: 13.06.2020 ----------------------------- #
				for experiment in self.hist["performance_history"]:
					epochs = int(experiment.split(":")[4])
					for threshold in self.hist["performance_history"][experiment]:
						for history_interval in self.hist["performance_history"][experiment][threshold]:
							val_count = epochs // int(history_interval)
							for entry in self.hist["performance_history"][experiment][threshold][history_interval]:
								for category in entry:
									for compression_mode in entry[category]:
										val = entry[category][compression_mode]
										if len(val) > val_count:
											entry[category][compression_mode] = val[:val_count]
											there_is_a_change = True
				# ------------------------------------------------ #

		if update_pruning_per_threshold:
			pass

		# Save the changes
		if there_is_a_change:
			self.save_data()
			print("Experiment logs are updated.")

	def delete_entry(self):
		"""
		Deletes the indicated erroneous entry from the logs
		"""
		# TODO: Implement this!
		pass

	def is_selected(self, experiment, mode=None):
		"""
		Checks the given experiment's settings to decide if it should be included in the reports
		"""
		fields = ["model", "optimizer", "loss", "lr", "epochs", "batch_size", "init_mode", "dataset", "l1_penalty"]
		settings = experiment.split(":")

		if mode is None:
			for i in range(len(settings)):
				if self.experiment_config[fields[i]] is not None and settings[i] != self.experiment_config[fields[i]]:
					return False

		elif mode == "performance":
			for i in range(len(settings)):
				if fields[i] != "dataset" \
						and self.experiment_config[fields[i]] is not None \
						and settings[i] != self.experiment_config[fields[i]]:
					return False

		return True

	def export_eig_stats_tex(self, hist, threshold="1e-04"):
		"""
			This function creates statictical table regarding real/complex eigenvalue distributions

			Note: Current version requires pre-determined threshold

			:param hist:
			:param threshold:
			:return:
		"""
		prefix = [
			"\\begin{table*}",
			"\t \\caption{CAPTION HERE}",
			"\t \\label{tab:eig_stats}",
			"\t \\centering",
			"\t \\resizebox{1\\textwidth}{!}{\\begin{tabular}{c|c|c|cc|cc|cc|cc}",
			"",
			"\t\t \\toprule",
			"\t\t & \\multirow{2}{*}{\\begin{tabular}{c}\\\\model\\end{tabular}} & \\multirow{2}{*}{\\begin{tabular}{c}\\\\total complex ratio\\end{tabular}} & \\multicolumn{2}{c|}{min\\_eig} & \\multicolumn{2}{c|}{min\\_eig\\_real} & \\multicolumn{2}{c|}{spectral\\_radius} & \\multicolumn{2}{c}{spectral\\_radius\\_real} \\\\",
			"\t\t \\cmidrule{4-11}",
			"\t\t & & & target & pruned & target & pruned & target & pruned & target & pruned \\\\"
		]
		datasets = ["CIFAR-10", "CIFAR-100", "tiny-imagenet"]
		models = ["MicroResNet32", "MicroResNet56", "MicroResNet110"]
		init_modes = ["random_init", "imagenet_init"]
		compression_modes = [
			"min_eig",
			"min_eig_real",
			"spectral_radius",
			"spectral_radius_real"
		]
		for model_mode in ["MicroResNet50", "else"]:
			table_data = {}

			for experiment in hist:
				if self.is_selected(experiment):
					info = experiment.split(":")
					model = info[0]
					init_mode = info[6]

					if model == model_mode or (model_mode == "else" and model != "MicroResNet50" and init_mode == "random_init"):
						dataset = info[7]
						label = model if model_mode == "else" else init_mode
						if dataset not in table_data:
							table_data[dataset] = {}
						if label not in table_data[dataset]:
							table_data[dataset][label] = {}

						experiment_data = {
							"total_complex_ratio": [],
							"target_complex_ratio": {c: [] for c in compression_modes},
							"pruned_complex_ratio": {c: [] for c in compression_modes}
						}
						target_complex_ratio_avg = {c: 0 for c in compression_modes}
						target_complex_ratio_std = {c: 0 for c in compression_modes}
						pruned_complex_ratio_avg = {c: 0 for c in compression_modes}
						pruned_complex_ratio_std = {c: 0 for c in compression_modes}

						if threshold in hist[experiment]:
							# Gather data
							for entry in hist[experiment][threshold]:
								temp = {
									"total_complex_list": 0,
									"total_real_list": 0,
									"target_complex_list": {c: 0 for c in compression_modes},
									"pruned_complex_list": {c: 0 for c in compression_modes},
									"target_real_list": {c: 0 for c in compression_modes},
									"pruned_real_list": {c: 0 for c in compression_modes},
								}
								layer_count = len(entry["layer_names"])

								for i in range(layer_count):
									for key in entry:
										if key == "total_complex_list" or key == "total_real_list":
											temp[key] += entry[key][i]
										elif key != "layer_names":
											for c in compression_modes:
												temp[key][c] += entry[key][i][c]

								# Process raw data
								total_complex_ratio = temp["total_complex_list"] / (temp["total_complex_list"] + temp["total_real_list"])
								experiment_data["total_complex_ratio"].append(total_complex_ratio)
								for c in compression_modes:
									target_complex_ratio = temp["target_complex_list"][c] / (temp["target_complex_list"][c] + temp["target_real_list"][c])
									experiment_data["target_complex_ratio"][c].append(target_complex_ratio)

									pruned_complex_ratio = temp["pruned_complex_list"][c] / (temp["pruned_complex_list"][c] + temp["pruned_real_list"][c])
									experiment_data["pruned_complex_ratio"][c].append(pruned_complex_ratio)

							# Compute statistics
							temp = np.array(experiment_data["total_complex_ratio"])
							total_complex_ratio_avg = np.average(temp)
							total_complex_ratio_std = np.std(temp)
							for c in compression_modes:
								temp = np.array(experiment_data["target_complex_ratio"][c])
								target_complex_ratio_avg[c] = np.average(temp)
								target_complex_ratio_std[c] = np.std(temp)

								temp = np.array(experiment_data["pruned_complex_ratio"][c])
								pruned_complex_ratio_avg[c] = np.average(temp)
								pruned_complex_ratio_std[c] = np.std(temp)

							# Store data
							table_data[dataset][label] = {
								"total_complex_ratio": {
									"avg": total_complex_ratio_avg,
									"std": total_complex_ratio_std
								},
								"target_complex_ratio": {
									"avg": target_complex_ratio_avg,
									"std": target_complex_ratio_std
								},
								"pruned_complex_ratio": {
									"avg": pruned_complex_ratio_avg,
									"std": pruned_complex_ratio_std
								}
							}

			# Prepare the table
			body = ["\n".join(prefix)]
			for i in range(len(datasets)):
				dataset = datasets[i]
				dataset_label = {"CIFAR-10": "C-10", "CIFAR-100": "C-100", "tiny-imagenet": "t-img"}[dataset]
				rows = 3 if model_mode == "else" else 2
				body.append("\t\t \\midrule")
				body.append("\t\t \\parbox[t]{2mm}{\\multirow{%d}{*}{\\rotatebox[origin=c]{90}{\\small{%s}}}}" % (rows, dataset_label))

				labels = models if model_mode == "else" else init_modes
				for j in range(len(labels)):
					label = labels[j]

					# Report total ratio
					val = "& -"
					try:
						avg_val = table_data[dataset][label]["total_complex_ratio"]["avg"]
						std_val = table_data[dataset][label]["total_complex_ratio"]["std"]
						val = "& $%.2f \\pm %.3f$ " % (avg_val, std_val)
					except Exception as e:
						pass

					new_line = "\t\t & %s %s " % (label.replace("_", "\\_"), val)

					# Report results per model/init
					for k in range(len(compression_modes)):
						c = compression_modes[k]

						for l in range(2):
							val = "& -"

							try:
								if l % 2 == 0:
									avg_val = table_data[dataset][label]["target_complex_ratio"]["avg"][c]
									std_val = table_data[dataset][label]["target_complex_ratio"]["std"][c]
								else:
									avg_val = table_data[dataset][label]["pruned_complex_ratio"]["avg"][c]
									std_val = table_data[dataset][label]["pruned_complex_ratio"]["std"][c]
								val = "& $%.2f \\pm %.3f$ " % (avg_val, std_val)

							except Exception as e:
								pass

							new_line += val

					new_line += "\\\\"
					body.append(new_line)

			body.append("\t\t \\bottomrule")
			body.append("\t\t \\end{tabular}}")
			body.append("\\end{table*}")

			# Export the LaTeX file
			file_name = "ThinMicroResNet" if model_mode == "else" else model_mode
			path = os.path.join(self.to_dir, "%s_eig_stats.tex" % file_name)
			with open(path, '+w') as tex_file:
				tex_file.write("\n".join(body))

	def plot_eig_analysis(self, hist):
		for experiment in hist:
			if self.is_selected(experiment):
				for threshold in hist[experiment]:
					print(hist[experiment][threshold].keys())
					for compression_mode in hist[experiment][threshold]:
						print("-", hist[experiment][threshold][compression_mode])
						#TODO: Complete the implementation!

	def auto_label_pruning(self, rects, vals):
		i = 0
		for rect in rects:
			height = rect.get_height()
			val = "%.2f%%" % vals[i]
			indicator = ">0" if val == "0.00%" and vals[i] > 0 else ""
			x_coord = rect.get_x() + rect.get_width() / 2.
			plt.text(x_coord, height + 0.1, val, ha="center", va="bottom", fontsize=12)
			plt.text(x_coord, height + 5, indicator, ha="center", va="bottom", fontsize=12)
			i += 1

	def plot_set_analysis(self, hist):
		try:
			for experiment in hist:
				if self.is_selected(experiment):
					for threshold in hist[experiment]:
						compression_mode_codes = hist[experiment][threshold][0]["codes"]
						entry_count = len(hist[experiment][threshold])
						to_file = os.path.join(self.to_dir, "%s_%s.pdf" % ("_".join(experiment.split(":")), threshold))

						# Accumulate data
						group_dict = {}
						for i in range(entry_count):
							temp_groups = hist[experiment][threshold][i]["groups"]
							for g in temp_groups:
								group_dict[g] = -1
						groups = sorted(list(group_dict.keys()))
						group_count = len(groups)
						for i in range(group_count):
							group_dict[groups[i]] = i

						sizes = np.zeros((group_count, entry_count))
						for i in range(entry_count):
							temp_groups = hist[experiment][threshold][i]["groups"]
							temp_sizes = hist[experiment][threshold][i]["group_sizes"]
							for j in range(len(temp_sizes)):
								sizes[group_dict[temp_groups[j]]][i] = temp_sizes[j]

						# Compute statistics
						sizes_avg = [0]*group_count
						sizes_std = [0]*group_count
						for i in range(group_count):
							sizes_avg[i] = np.average(sizes[i])
							sizes_std[i] = np.std(sizes[i])

						# Plot the bar chart
						bar_width = 0.35
						opacity = 0.8
						fig = plt.figure(figsize=(13, 6))
						ax1 = fig.add_subplot(1, 1, 1)
						y_pos = np.arange(group_count)
						rects = ax1.bar(y_pos, sizes_avg, bar_width, alpha=opacity, color=self.tableau20[:group_count], yerr=sizes_std, capsize=5)
						self.auto_label_pruning(rects, sizes_avg)
						temp = experiment.split(":")
						model_name = temp[0][5:]
						init_mode = temp[6]
						dataset = temp[7]
						ax1.set_title("%s - %s - %s" % (model_name, dataset, init_mode))
						ax1.set_xticks(y_pos)
						ax1.set_xticklabels(groups, rotation=33)
						ax1.set_ylabel('Pruned/Total param ratio (%)')
						ax1.set_ylim(ymin=0, ymax=100)
						ax1.grid(linestyle=":", linewidth=1, alpha=opacity)
						ax1.tick_params(axis="x", labelsize=9)

						# Custom legend
						keys = sorted(list(compression_mode_codes.keys()))
						legend_elements = []
						for key in keys:
							label = "%s : %s" % (key, compression_mode_codes[key])
							legend_elements.append(Line2D([0], [0], color='black', lw=1, label=label))
						ax1.legend(handles=legend_elements, handlelength=0)

						# Save the diagram
						plt.tight_layout()
						plt.grid(True)
						plt.savefig(to_file, bbox_inches="tight")
						plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_pruning_per_threshold(self, hist):
		try:
			compression_modes = {
				"min_eig": "darkolivegreen",
				"min_eig_real": "limegreen",
				"det": "#005f87",
				"det_corr": "blue",
				"spectral_radius": "red",
				"spectral_radius_real": "crimson",
				"spectral_norm": "darkorange",
				"weight": "magenta"
			}

			for experiment in hist:
				if self.is_selected(experiment):
					to_file = os.path.join(self.to_dir, "%s.png" % ("_".join(experiment.split(":"))))
					thresholds = hist[experiment][0]["thresholds"]
					threshold_count = len(thresholds)
					entry_count = len(hist[experiment])
					temp_vals = {key: np.zeros((threshold_count, entry_count)) for key in hist[experiment][0]["final_vals"].keys()}
					final_vals_avg = {key: np.zeros(threshold_count) for key in hist[experiment][0]["final_vals"].keys()}
					final_vals_std = {key: np.zeros(threshold_count) for key in hist[experiment][0]["final_vals"].keys()}

					# Accumulate data
					for i in range(entry_count):
						entry = hist[experiment][i]
						temp = entry["final_vals"]
						for compression_mode in temp:
							for j in range(threshold_count):
								temp_vals[compression_mode][j][i] = temp[compression_mode][j]

					# Compute statistics
					for compression_mode in temp_vals:
						for i in range(threshold_count):
							final_vals_avg[compression_mode][i] = np.average(temp_vals[compression_mode][i])
							final_vals_std[compression_mode][i] = np.std(temp_vals[compression_mode][i])

					# Draw the chart
					indices = np.arange(len(thresholds))
					fig = plt.figure(figsize=(18, 9))
					shader_alpha = 0.3
					for compression_mode in compression_modes:
						label = "det_gram" if compression_mode == "det_corr" else compression_mode
						plt.plot(indices, final_vals_avg[compression_mode], color=compression_modes[compression_mode], label=label, linewidth=2)
						plt.fill_between(indices,
										 final_vals_avg[compression_mode] + final_vals_std[compression_mode],
										 final_vals_avg[compression_mode] - final_vals_std[compression_mode],
										 color=compression_modes[compression_mode],
										 alpha=shader_alpha)

					plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useOffset=False)
					plt.xlabel("Significance Threshold", fontsize=10)
					plt.xticks(indices, thresholds)
					plt.ylabel("Ratio of pruned/total kernels (%)", fontsize=10)
					plt.ylim(ymin=0, ymax=100)
					plt.legend(loc="upper left")
					plt.grid(linestyle=":", linewidth=1, alpha=0.8)

					# Save the figure
					plt.tight_layout()
					plt.grid(True)
					plt.savefig(to_file, bbox_inches="tight")
					plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_pruning_per_threshold_miniature(self, hist):
		try:
			datasets = {"CIFAR-10": 0, "CIFAR-100": 1, "tiny-imagenet": 2}
			models = {"MicroResNet32": 0, "MicroResNet56": 1, "MicroResNet110": 2, "MicroResNet50": None}
			init_modes = {"static_init": 0, "random_init": 1, "imagenet_init": 2}
			compression_modes = {
				"min_eig": "darkolivegreen",
				"min_eig_real": "limegreen",
				"det": "#005f87",
				"det_corr": "blue",
				"spectral_radius": "red",
				"spectral_radius_real": "crimson",
				"spectral_norm": "darkorange",
				"weight": "magenta"
			}
			shader_alpha = 0.3
			linewidth = 1
			row_count = 3

			for model_mode in ["MicroResNet50", "else"]:
				fig, axes = plt.subplots(3, 3, figsize=(12, 8))

				for experiment in hist:
					if self.is_selected(experiment):
						info = experiment.split(":")
						model = info[0]
						init_mode = info[6]

						if model == model_mode or (model_mode == "else" and model != "MicroResNet50" and init_mode == "random_init"):
							dataset = info[7]
							thresholds = hist[experiment][0]["thresholds"]
							threshold_count = len(thresholds)
							entry_count = len(hist[experiment])
							temp_vals = {key: np.zeros((threshold_count, entry_count)) for key in hist[experiment][0]["final_vals"].keys()}
							final_vals_avg = {key: np.zeros(threshold_count) for key in hist[experiment][0]["final_vals"].keys()}
							final_vals_std = {key: np.zeros(threshold_count) for key in hist[experiment][0]["final_vals"].keys()}

							# Accumulate data
							for i in range(entry_count):
								entry = hist[experiment][i]
								temp = entry["final_vals"]
								for compression_mode in temp:
									for j in range(threshold_count):
										temp_vals[compression_mode][j][i] = temp[compression_mode][j]

							# Compute statistics
							for compression_mode in temp_vals:
								for i in range(threshold_count):
									final_vals_avg[compression_mode][i] = np.average(temp_vals[compression_mode][i])
									final_vals_std[compression_mode][i] = np.std(temp_vals[compression_mode][i])

							# Draw the chart
							col = datasets[dataset]
							row = init_modes[init_mode] if model_mode == "MicroResNet50" else models[model]
							indices = np.arange(len(thresholds))

							for compression_mode in compression_modes:
								label = "det_gram" if compression_mode == "det_corr" else compression_mode
								axes[row, col].plot(indices, final_vals_avg[compression_mode], color=compression_modes[compression_mode], linestyle="--", marker=".", label=label, linewidth=linewidth, alpha=shader_alpha)
								axes[row, col].fill_between(indices,
															final_vals_avg[compression_mode] + final_vals_std[compression_mode],
															final_vals_avg[compression_mode] - final_vals_std[compression_mode],
															color=compression_modes[compression_mode],
															alpha=shader_alpha)

							axes[row, col].set_ylim(ymin=0, ymax=100)
							axes[row, col].grid(linestyle=":", linewidth=0.5, alpha=0.6)
							temp = [to_scientific(thresholds[i]) if i % 4 == 0 else " " for i in range(len(thresholds))]
							axes[row, col].set_xticks(indices)
							axes[row, col].set_xticklabels(temp)
							axes[row, col].axvline(x=16, linewidth=1, linestyle="--", color="crimson")
							if col == 0:
								ylabel = init_mode if model_mode == "MicroResNet50" else model[5:]
								axes[row, col].set_ylabel("%s - pruning ratio (%s)" % (ylabel, "%"))
							if row == 0:
								axes[row, col].set_title(dataset)
							elif row == row_count - 1:
								axes[row, col].set_xlabel("significance threshold")

				# Custom legend
				legend_elements = []
				labels = []
				for c in compression_modes:
					temp = "det_gram" if c == "det_corr" else c
					labels.append(temp)
					legend_elements.append(Line2D([0], [0], color=compression_modes[c], alpha=shader_alpha, lw=9, label=c))
				fig.legend(handles=legend_elements, labels=labels, handlelength=1, loc="lower center", bbox_to_anchor=(0.5, -0.013), ncol=8, frameon=False)
				fig.subplots_adjust(bottom=0.02)

				# Save the figure
				file_name = "ThinMicroResNet" if model_mode == "else" else model_mode
				to_file = os.path.join(self.to_dir, "%s_pruning_per_threshold.pdf" % file_name)
				plt.tight_layout()
				plt.savefig(to_file, bbox_inches="tight")
				plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)
			import traceback
			traceback.print_exc()

	def plot_pruning_per_layer(self, hist):
		try:
			for experiment in hist:
				if self.is_selected(experiment):
					for threshold in hist[experiment]:
						for compression_mode in hist[experiment][threshold]:
							to_file = os.path.join(self.to_dir, "%s_%s_%s.png" % ("_".join(experiment.split(":")), threshold, compression_mode))
							layer_names = hist[experiment][threshold][compression_mode][0]["layer_names"]
							total_params = [float(s) for s in hist[experiment][threshold][compression_mode][0]["total_params"]]
							active_params_percentage_dict = {layer: [] for layer in layer_names}
							active_params_percentage_avg = []
							active_params_percentage_std = []
							active_params_dict = {layer: [] for layer in layer_names}
							active_params_avg = []
							active_params_std = []
							layer_count = len(layer_names)

							# Gather multiple results for each layer
							for entry in hist[experiment][threshold][compression_mode]:
								for i in range(layer_count):
									active_params_percentage_dict[layer_names[i]].append(float(entry["active_params"][i]) * 100.0 / total_params[i])
									active_params_dict[layer_names[i]].append(float(entry["active_params"][i]))

							# Compute statistics per layer
							for i in range(layer_count):
								temp = np.array(active_params_percentage_dict[layer_names[i]])
								active_params_percentage_avg.append(np.average(temp))
								active_params_percentage_std.append(np.std(temp))

								temp = np.array(active_params_dict[layer_names[i]])
								active_params_avg.append(np.average(temp))
								active_params_std.append(np.std(temp))

							# Start plotting
							fig = plt.figure(figsize=(23, 11))
							bar_width = 0.35
							opacity = 0.8

							# First, build active params vs. total params per layer chart
							ax1 = fig.add_subplot(2, 1, 1)
							y_pos = np.arange(len(layer_names))
							ax1.bar(y_pos,
									active_params_avg,
									bar_width,
									alpha=opacity,
									color="#D62728",
									label='Active',
									edgecolor="#D62728",
									yerr=active_params_std,
									capsize=5,
									hatch="")
							ax1.bar(y_pos + bar_width,
									total_params,
									bar_width,
									alpha=opacity,
									color="#005F87",
									label='Total',
									edgecolor="#004669",
									hatch="//")

							ax1.set_xticks(y_pos + 0.5 * bar_width)
							ax1.set_xticklabels(layer_names, rotation=33)
							ax1.set_ylabel('# of params')
							ax1.legend(loc="upper left")
							ax1.grid(linestyle=":", linewidth=1, alpha=opacity)
							ax1.tick_params(axis="x", labelsize=9)

							# Second, build active/total params per layer chart
							ax2 = fig.add_subplot(2, 1, 2)
							ax2.bar(y_pos,
									active_params_percentage_avg,
									bar_width,
									alpha=opacity,
									color="#D62728",
									edgecolor="#D62728",
									yerr=active_params_percentage_std,
									capsize=5,
									hatch="//")

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

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_pruning_per_layer_miniature(self, hist, threshold="1e-04"):
		"""
		This function plots bar chart of layer-by-layer compression info for multiple compression modes

		Note: Current version requires pre-determined threshold to plot

		:param hist:
		:param threshold:
		:return:
		"""
		try:
			datasets = {"CIFAR-10": 0, "CIFAR-100": 1, "tiny-imagenet": 2}
			models = {"MicroResNet32": 0, "MicroResNet56": 1, "MicroResNet110": 2, "MicroResNet50": None}
			init_modes = {"static_init": 0, "random_init": 1, "imagenet_init": 2}
			compression_modes = {
				"min_eig_real": 0,
				"min_eig": 1,
				"det": 2,
				"det_corr": 3,
				"spectral_radius_real": 4,
				"spectral_radius": 5,
				"weight": 6,
				"spectral_norm": 7
			}

			for model_mode in ["MicroResNet50", "else"]:
				fig = plt.figure(figsize=(30, 13))
				row_count = 3

				for experiment in hist:
					if self.is_selected(experiment):
						info = experiment.split(":")
						model = info[0]
						init_mode = info[6]

						if model == model_mode or (model_mode == "else" and model != "MicroResNet50" and init_mode == "random_init"):
							dataset = info[7]

							if threshold in hist[experiment]:
								# Accumulate data across compression modes for 3D surface plot
								data = {"x": [], "y": [], "z": []}

								for compression_mode in compression_modes:
									layer_names = hist[experiment][threshold][compression_mode][0]["layer_names"]
									total_params = [float(s) for s in hist[experiment][threshold][compression_mode][0]["total_params"]]
									val_dict = {layer: [] for layer in layer_names}
									layer_count = len(layer_names)

									# Gather multiple results for each layer
									for entry in hist[experiment][threshold][compression_mode]:
										for i in range(layer_count):
											val = float(entry["active_params"][i]) * 100.0 / total_params[i]
											val_dict[layer_names[i]].append(val)

									# Compute statistics per layer and store as 3D data
									y = compression_modes[compression_mode]
									for i in range(layer_count):
										temp = np.array(val_dict[layer_names[i]])
										data["x"].append(i)
										data["y"].append(y)
										data["z"].append(np.average(temp))

								# Create subplot
								col = datasets[dataset]
								row = init_modes[init_mode] if model_mode == "MicroResNet50" else models[model]
								ax = fig.add_subplot(3, 3, row*3 + col + 1, projection='3d')
								ax.plot_trisurf(data["x"], data["y"], data["z"], cmap=plt.cm.coolwarm, linewidth=0.02)
								ax.set_zlim(bottom=0, top=100)
								labels = ["det_gram" if c == "det_corr" else c for c in compression_modes]
								ax.set_yticks(np.arange(len(labels)))
								ax.set_yticklabels(labels,
												   verticalalignment="baseline",
												   horizontalalignment="left")
								ax.set_zlabel("Active/Total param ratio (%)")
								if row == 0:
									ax.set_title(dataset)
								elif row == row_count - 1:
									ax.set_xlabel("Layer #")
								ylabel = init_mode if model_mode == "MicroResNet50" else model[5:]
								ax.text(1.1, 0.5, 0.5, ylabel, size=13, verticalalignment="center")
								ax.view_init(30, 272)

				file_name = "ThinMicroResNet" if model_mode == "else" else model_mode
				to_file = os.path.join(self.to_dir, "%s_pruning_per_layer.pdf" % file_name)
				fig.subplots_adjust(wspace=0)
				plt.grid(False)
				plt.tight_layout()
				plt.savefig(to_file, bbox_inches="tight", pad_inches=0)
				plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)
			import traceback
			traceback.print_exc()

	def export_performance_tex(self, results, datasets):
		prefix = [
			"\\begin{table*}",
			"\t \\caption{CAPTION HERE}",
			"\t \\label{tab:performance}",
			"\t \\centering",
			"\t \\resizebox{1\\textwidth}{!}{\\begin{tabular}{c|l|cc|cc|cc|ccc}",
			"",
			"\t\t \\toprule",
			"\t\t & \\multirow{2}{*}{\\begin{tabular}{c}\\\\compression mode\\end{tabular}} & \\multicolumn{2}{c|}{ResNet32} & \\multicolumn{2}{c|}{ResNet56} & \\multicolumn{2}{c|}{ResNet110} & \\multicolumn{3}{c}{ResNet50} \\\\",
			"\t\t \\cmidrule{3-11}",
			"\t\t & & static & random & static & random & static & random & static & random & imagenet \\\\"
		]
		models = ["MicroResNet32", "MicroResNet56", "MicroResNet110", "MicroResNet50"]
		categories = ["param_count", "score", "accuracy"]
		init_modes = ["static_init", "random_init", "imagenet_init"]
		compression_modes = [
			"min_eig",
			"min_eig_real",
			"det",
			"det_corr",
			"spectral_radius",
			"spectral_radius_real",
			"spectral_norm",
			"weight"
		]
		num_thresholds = len(list(results.keys()))

		for threshold in results:
			for category in categories:
				# Determine winners
				winners = np.zeros((3, 9))
				data = np.zeros((3, 9, 8))
				for i in range(len(datasets)):
					for j in range(len(compression_modes)):
						compression_mode = compression_modes[j]
						for k in range(len(models)):
							model = models[k]
							columns = 3 if model == "MicroResNet50" else 2

							for l in range(columns):
								init_mode = init_modes[l]

								try:
									data[i][k * 2 + l][j] = results[threshold][model][init_mode][category]["avg"][compression_mode][i]
								except Exception as e:
									pass
				for i in range(3):
					for j in range(9):
						winners[i][j] = np.argmax(data[i][j])

				# Prepare the table with indicated winners
				body = ["\n".join(prefix)]
				for i in range(len(datasets)):
					dataset = datasets[i]
					body.append("\t\t \\midrule")
					body.append("\t\t \\parbox[t]{2mm}{\\multirow{8.5}{*}{\\rotatebox[origin=c]{90}{%s}}}" % dataset)

					# Include baseline for accuracy table
					if category == "accuracy":
						new_line = "\t\t & None "

						for k in range(len(models)):
							model = models[k]
							columns = 3 if model == "MicroResNet50" else 2

							for l in range(columns):
								init_mode = init_modes[l]

								val = "& -"
								try:
									avg_val = results[threshold][model][init_mode][category]["avg"]["None"][i]
									std_val = results[threshold][model][init_mode][category]["std"]["None"][i]
									val = "& $%.4f \\pm %.3f$ " % (avg_val, std_val)
								except Exception as e:
									pass

								new_line += val
						new_line += "\\\\"
						body.append(new_line)
						body.append("\t\t \\cmidrule{2-11}")

					# Report results per compression mode
					for j in range(len(compression_modes)):
						compression_mode = compression_modes[j]
						label = "det_gram" if compression_mode == "det_corr" else compression_mode
						new_line = "\t\t & %s " % label.replace("_", "\\_")

						for k in range(len(models)):
							model = models[k]
							columns = 3 if model == "MicroResNet50" else 2

							for l in range(columns):
								init_mode = init_modes[l]

								val = "& -"
								try:
									avg_val = results[threshold][model][init_mode][category]["avg"][compression_mode][i]
									std_val = results[threshold][model][init_mode][category]["std"][compression_mode][i]
									if winners[i][k * 2 + l] == j:
										val = "& $\\textbf{%.4f} \\pm %.3f$ " % (avg_val, std_val)
									else:
										val = "& $%.4f \\pm %.3f$ " % (avg_val, std_val)
								except Exception as e:
									pass

								new_line += val
						new_line += "\\\\"
						body.append(new_line)

				body.append("\t\t  \\bottomrule")
				body.append("\t\t  \\end{tabular}}")
				body.append("\t \\vspace{-3pt}")
				body.append("\\end{table*}")
				path = os.path.join(self.to_dir, "performance_%s_%s.tex" % (threshold, category)) if num_thresholds > 1 else os.path.join(self.to_dir, "performance_%s.tex" % category)

				# Export the LaTeX file
				with open(path, '+w') as tex_file:
					tex_file.write("\n".join(body))

	def plot_performance(self, hist, export_png=True, export_tex=True):
		try:
			experiments = {}

			# Preprocessing for performance chart format
			for experiment in hist:
				if self.is_selected(experiment, mode="performance"):
					temp = experiment.split(":")
					del temp[7]
					model_name = ":".join(temp)
					if model_name not in experiments:
						experiments[model_name] = {}

					for threshold in hist[experiment]:
						if float(threshold) == self.experiment_config["threshold"] or self.experiment_config["threshold"] is None:
							if threshold not in experiments[model_name]:
								experiments[model_name][threshold] = []
							experiments[model_name][threshold].append(experiment)

			# Accumulate data and plot the charts
			summary = {}
			for model_name in experiments:
				for threshold in experiments[model_name]:
					if threshold not in summary:
						summary[threshold] = {}

					datasets = []
					results = {"param_count": {"avg": {}, "std": {}},
							   "score": {"avg": {}, "std": {}},
							   "accuracy": {"avg": {}, "std": {}}}
					for experiment in sorted(experiments[model_name][threshold]):
						dataset = experiment.split(":")[7]
						datasets.append(dataset)

						# Accumulate data for each chart type separately
						temp_dict = {"param_count": {},
									 "score": {},
									 "accuracy": {}}
						for entry in hist[experiment][threshold]:
							for compression_mode in entry:
								for category in entry[compression_mode]:
									if category in temp_dict:
										if compression_mode in temp_dict[category]:
											temp_dict[category][compression_mode].append(entry[compression_mode][category])
										else:
											temp_dict[category][compression_mode] = [entry[compression_mode][category]]

						for category in temp_dict:
							# Convert param_count to compression ratio
							if category == "param_count":
								original_param_count = temp_dict[category]["None"][0]

							# STD computation to plot confidence intervals
							for compression_mode in temp_dict[category]:
								if compression_mode not in results[category]["avg"]:
									results[category]["avg"][compression_mode] = []
								if compression_mode not in results[category]["std"]:
									results[category]["std"][compression_mode] = []

								temp_array = 1 - (np.array(temp_dict[category][compression_mode]) / original_param_count) if category == "param_count" else np.array(temp_dict[category][compression_mode])
								results[category]["avg"][compression_mode].append(np.average(temp_array))
								results[category]["std"][compression_mode].append(np.std(temp_array))

					if export_png:
						ylabels = {"param_count": "Compression ratio", "score": "Compression score", "accuracy": "Accuracy"}
						ymins = {"param_count": 0, "score": 0.85, "accuracy": 0.3}
						for category in results:
							# Plot the chart
							fig = plt.figure(figsize=(23, 11))
							to_file = os.path.join(self.to_dir, "%s_%s_%s.png" % ("_".join(model_name.split(":")), threshold, category))

							# Chart data
							y_pos = np.arange(len(datasets))

							# Chart configuration
							bar_width = 0.04
							opacity = 0.8
							colors = [self.tableau20[6], "#FFBE7D", self.tableau20[5], "#00ff80", "#BAB0AC", self.tableau20[1], "#17BECF", "#99B3E6", "#BF8040"]
							edgecolors = [self.tableau20[6], self.tableau20[7], self.tableau20[4], "#006666", (0,0,0), self.tableau20[0], "#1F77B4", "#3366CC", "#CC7A00"]

							i = 0
							for compression_mode in results[category]["avg"]:
								if compression_mode != "None":
									plt.bar(y_pos + bar_width * i,
											results[category]["avg"][compression_mode],
											bar_width,
											alpha=opacity,
											color=colors[i],
											label=compression_mode,
											edgecolor=edgecolors[i],
											yerr=results[category]["std"][compression_mode],
											capsize=5,
											hatch=self.hatches[i])
									i += 1

							# Draw baselines
							if category == "accuracy":
								thresholds = results[category]["avg"]["None"]
								for i in range(len(thresholds)):
									plt.plot([i-0.13, i+0.43], [thresholds[i], thresholds[i]], "k--", color="crimson")

							plt.xticks(y_pos + 3 * bar_width, datasets)
							plt.xlabel("Dataset")
							plt.ylabel(ylabels[category])
							#plt.ylim(ymin=ymins[category])
							plt.legend(fontsize=11)
							plt.grid(True)
							plt.tight_layout()
							plt.savefig(to_file, bbox_inches="tight")
							plt.close(fig)

					if export_tex:
						#TODO: Current version assumes & supports variations in {model, init_mode, threshold}, so NOT complete yet!!
						temp = model_name.split(":")
						init_mode = temp[6]
						model = temp[0]
						if model in summary[threshold]:
							summary[threshold][model][init_mode] = results
						else:
							summary[threshold][model] = {init_mode: results}

			self.export_performance_tex(summary, datasets)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_performance_history_winners(self, hist, threshold="1e-04", history_interval="10"):
		"""
		This function plots line chart of compression scores through epochs for multiple compression modes

		Note: Current version requires pre-determined threshold & history_interval to plot

		:param hist:
		:param threshold:
		:param history_interval:
		:return:
		"""
		try:
			param_count = {
				"MicroResNet32": {
					"CIFAR-10": 470218,
					"CIFAR-100": 476068,
					"tiny-imagenet": 520968
				},
				"MicroResNet56": {
					"CIFAR-10": 861770,
					"CIFAR-100": 867620,
					"tiny-imagenet": 912520
				},
				"MicroResNet110": {
					"CIFAR-10": 1742762,
					"CIFAR-100": 1748612,
					"tiny-imagenet": 1793512
				}
				,"MicroResNet50": {
					"CIFAR-10": 23608202,
					"CIFAR-100": 23792612,
					"tiny-imagenet": 23997512
				}
			}
			datasets = {"CIFAR-10": 0, "CIFAR-100": 1, "tiny-imagenet": 2}
			models = {"MicroResNet32": 0, "MicroResNet56": 1, "MicroResNet110": 2, "MicroResNet50": None}
			categories = ["accuracy", "param_count", "score"]
			init_modes = {"random_init": 0, "imagenet_init": 1}
			compression_modes = {
				"min_eig": "darkolivegreen",
				"min_eig_real": "limegreen",
				"det": "#005f87",
				"det_corr": "blue",
				"spectral_radius": "red",
				"spectral_radius_real": "crimson",
				"spectral_norm": "darkorange",
				"weight": "magenta"
			}
			shader_alpha = 0.3
			int_history_interval = int(history_interval)

			for category in categories:
				for model_mode in["MicroResNet50", "else"]:
					row_count = 2 if model_mode == "MicroResNet50" else 3
					fig, axes = plt.subplots(row_count, 3, figsize=(15, row_count * 2))

					# Preprocessing for performance chart format
					for experiment in hist:
						if self.is_selected(experiment):
							info = experiment.split(":")
							model = info[0]

							if model == model_mode or (model_mode == "else" and model != "MicroResNet50"):
								init_mode = info[6]
								dataset = info[7]
								val_count = int(info[4]) // int_history_interval
								avg_dict = {c: np.zeros(val_count) for c in compression_modes}
								std_dict = {c: np.zeros(val_count) for c in compression_modes}

								# Process data for each experiment = (model, dataset, init_mode)
								if threshold in hist[experiment]:
									if history_interval in hist[experiment][threshold]:
										for compression_mode in compression_modes:
											for i in range(val_count):
												temp_list = []
												for entry in hist[experiment][threshold][history_interval]:
													val = entry[category][compression_mode][i]
													temp_list.append(val)
												temp_array = 1 - (np.array(temp_list) / param_count[model][dataset]) if category == "param_count" else np.array(temp_list)
												avg_dict[compression_mode][i] = np.average(temp_array)
												std_dict[compression_mode][i] = np.std(temp_array)

								if init_mode in init_modes:
									final_vals = np.zeros(val_count)
									colors = [None] * val_count
									col = datasets[dataset]
									row = init_modes[init_mode] if model_mode == "MicroResNet50" else models[model]

									# Indentify best modes and their corresponding colors for the area chart
									for i in range(val_count):
										max_val = -1
										for c in compression_modes:
											if max_val < avg_dict[c][i] or (max_val == avg_dict[c][i] and i > 0 and colors[i - 1] == compression_modes[c]):
												max_val = avg_dict[c][i]
												colors[i] = compression_modes[c]
										final_vals[i] = max_val

									# Plot subcharts
									temp = np.arange(val_count)
									axes[row, col].plot(final_vals, alpha=0.5)
									for i in range(val_count-1):
										axes[row, col].fill_between([i, i+1],
																	[final_vals[i], final_vals[i+1]],
																	0,
																	color=colors[i],
																	alpha=shader_alpha)
									if col == 0:
										ylabel = init_mode if model_mode == "MicroResNet50" else model[5:]
										axes[row, col].set_ylabel("%s\n%s" % (ylabel, category))
									if row == 0:
										axes[row, col].set_title(dataset)
									elif row == row_count - 1:
										axes[row, col].set_xlabel("epochs")
									axes[row, col].set(xticks=temp, xticklabels=[str(i * int_history_interval) if i % 3 == 0 else "" for i in temp])
									axes[row, col].set_ylim(ymin=0, ymax=1)

					# Custom legend
					legend_elements = []
					labels = []
					for c in compression_modes:
						label = "det_gram" if c == "det_corr" else c
						labels.append(label)
						legend_elements.append(Line2D([0], [0], color=compression_modes[c], alpha=shader_alpha, lw=9, label=c))
					fig.legend(handles=legend_elements, labels=labels, handlelength=1, loc="lower center", bbox_to_anchor=(0.5, -0.013), ncol=8, frameon=False)
					fig.subplots_adjust(bottom=0.02)

					file_name = "ThinMicroResNet" if model_mode == "else" else model_mode
					to_file = os.path.join(self.to_dir, "%s_%s_history_winners.pdf" % (file_name, category))
					#plt.legend(fontsize=11)
					plt.grid(False)
					plt.tight_layout()
					plt.savefig(to_file, bbox_inches="tight")
					plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_performance_history_full(self, hist, threshold="1e-04", history_interval="10"):
		"""
		This function plots line chart of compression scores through epochs for multiple compression modes

		Note: Current version requires pre-determined threshold & history_interval to plot

		:param hist:
		:param threshold:
		:param history_interval:
		:return:
		"""
		try:
			param_count = {
				"MicroResNet32": {
					"CIFAR-10": 470218,
					"CIFAR-100": 476068,
					"tiny-imagenet": 520968
				},
				"MicroResNet56": {
					"CIFAR-10": 861770,
					"CIFAR-100": 867620,
					"tiny-imagenet": 912520
				},
				"MicroResNet110": {
					"CIFAR-10": 1742762,
					"CIFAR-100": 1748612,
					"tiny-imagenet": 1793512
				}
				,"MicroResNet50": {
					"CIFAR-10": 23608202,
					"CIFAR-100": 23792612,
					"tiny-imagenet": 23997512
				}
			}
			datasets = {"CIFAR-10": 0, "CIFAR-100": 1, "tiny-imagenet": 2}
			models = {"MicroResNet32": 0, "MicroResNet56": 1, "MicroResNet110": 2, "MicroResNet50": None}
			categories = ["accuracy", "param_count", "score"]
			init_modes = {"random_init": 0, "imagenet_init": 1}
			compression_modes = {
				"min_eig": "darkolivegreen",
				"min_eig_real": "limegreen",
				"det": "#005f87",
				"det_corr": "blue",
				"spectral_radius": "red",
				"spectral_radius_real": "crimson",
				"spectral_norm": "darkorange",
				"weight": "magenta"
			}
			linewidth = 2
			shader_alpha = 0.3
			int_history_interval = int(history_interval)

			for category in categories:
				for model_mode in["MicroResNet50", "else"]:
					row_count = 2 if model_mode == "MicroResNet50" else 3
					fig, axes = plt.subplots(row_count, 3, figsize=(15, row_count * 2))

					# Preprocessing for performance chart format
					for experiment in hist:
						if self.is_selected(experiment):
							info = experiment.split(":")
							model = info[0]

							if model == model_mode or (model_mode == "else" and model != "MicroResNet50"):
								init_mode = info[6]
								dataset = info[7]
								val_count = int(info[4]) // int_history_interval
								avg_dict = {c: np.zeros(val_count) for c in compression_modes}
								std_dict = {c: np.zeros(val_count) for c in compression_modes}

								# Process data for each experiment = (model, dataset, init_mode)
								if threshold in hist[experiment]:
									if history_interval in hist[experiment][threshold]:
										for compression_mode in compression_modes:
											for i in range(val_count):
												temp_list = []
												for entry in hist[experiment][threshold][history_interval]:
													val = entry[category][compression_mode][i]
													temp_list.append(val)
												temp_array = 1 - (np.array(temp_list) / param_count[model][dataset]) if category == "param_count" else np.array(temp_list)
												avg_dict[compression_mode][i] = np.average(temp_array)
												std_dict[compression_mode][i] = np.std(temp_array)

								if init_mode in init_modes:
									# Draw the chart
									col = datasets[dataset]
									row = init_modes[init_mode] if model_mode == "MicroResNet50" else models[model]
									indices = np.arange(val_count)

									for c in compression_modes:
										label = "det_gram" if c == "det_corr" else c
										axes[row, col].plot(indices, avg_dict[c], color=compression_modes[c], linestyle="--",  marker=".", label=label, linewidth=linewidth, alpha=shader_alpha)

										axes[row, col].fill_between(indices,
																	avg_dict[c] + std_dict[c],
																	avg_dict[c] - std_dict[c],
																	color=compression_modes[c],
																	alpha=shader_alpha)


									if col == 0:
										ylabel = init_mode if model_mode == "MicroResNet50" else model[5:]
										axes[row, col].set_ylabel("%s\n%s" % (ylabel, category))
									if row == 0:
										axes[row, col].set_title(dataset)
									elif row == row_count - 1:
										axes[row, col].set_xlabel("epochs")
									axes[row, col].set(xticks=indices, xticklabels=[str(i * int_history_interval) if i % 3 == 0 else "" for i in indices])
									axes[row, col].set_ylim(ymin=0, ymax=1)
									axes[row, col].grid(linestyle=":", linewidth=0.5, alpha=0.6)

					# Custom legend
					legend_elements = []
					labels = []
					for c in compression_modes:
						label = "det_gram" if c == "det_corr" else c
						labels.append(label)
						legend_elements.append(Line2D([0], [0], color=compression_modes[c], alpha=shader_alpha, lw=9, label=c))
					fig.legend(handles=legend_elements, labels=labels, handlelength=1, loc="lower center", bbox_to_anchor=(0.5, -0.013), ncol=8, frameon=False)
					fig.subplots_adjust(bottom=0.02)

					file_name = "ThinMicroResNet" if model_mode == "else" else model_mode
					to_file = os.path.join(self.to_dir, "%s_%s_history_full.pdf" % (file_name, category))
					#plt.legend(fontsize=11)
					plt.tight_layout()
					plt.savefig(to_file, bbox_inches="tight")
					plt.close(fig)

		except Exception as e:
			print("Missing key!")
			print("-----------------------------------------")
			print(e)

	def plot_learning_curve(self, hist):
		pass

	def run(self):

		for chart_type in self.hist:
			print("Plotting...", chart_type)

			if chart_type == "eig_stats":
				self.export_eig_stats_tex(self.hist["eig_stats"])

			#elif chart_type == "eig_analysis":
			#	self.plot_eig_analysis(self.hist["eig_analysis"])

			elif chart_type == "set_analysis":
				self.plot_set_analysis(self.hist["set_analysis"])

			#elif chart_type == "pruning_per_threshold":
			#	self.plot_pruning_per_threshold(self.hist["pruning_per_threshold"])
			#	self.plot_pruning_per_threshold_miniature(self.hist["pruning_per_threshold"])

			#elif chart_type == "pruning_per_layer":
			#	self.plot_pruning_per_layer(self.hist["pruning_per_layer"])
			#	self.plot_pruning_per_layer_miniature(self.hist["pruning_per_layer"])

			#elif chart_type == "performance":
			#	self.plot_performance(self.hist["performance"], export_png=False)

			#elif chart_type == "performance_history":
			#	self.plot_performance_history_winners(self.hist["performance_history"])
			#	self.plot_performance_history_full(self.hist["performance_history"])

			elif chart_type == "learning_curve":
				self.plot_learning_curve(self.hist["learning_curve"])

		print("Done.")

if __name__ == '__main__':
	experimentProcessor = ExperimentProcessor(path="C:\\Users\\muare\\Desktop\\experiment_ALL_2020_09_15.json", to_dir="C:\\Users\\muare\\Desktop\\CompressionResults")
	experimentProcessor.update_logs(update_set_analysis=True, update_performance_history=True)
	experimentProcessor.run()

	#experimentProcessor.merge_logs(path="C:\\Users\\muare\\Desktop\\experiment_tiny_imagenet_7.json")
