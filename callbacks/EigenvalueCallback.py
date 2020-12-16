"""
Title           :EigenvalueCallback.py
Description     :Custom callback to store eigenvalues of kernels during training
Author          :Ilke Cugu
Date Created    :05-07-2019
Date Modified   :16-06-2020
version         :2.6.6
python_version  :3.6.6
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from matplotlib.lines import Line2D
from layers.MicroConv2D import MicroConv2D

class EigenvalueCallback(keras.callbacks.Callback):
	"""
	Callback to track the changes in eigenvalues of the kernels in a given MicroResNet

	# Arguments
		:param model_name: (string) model name for the resulting charts
		:param to_dir: (string) dir path to save the charts
		:param experiment_recorder: (ExperimentRecorder) for cumulative logging of empirical results
	"""
	def __init__(self, model_name="MicroResNet", to_dir="", experiment_recorder=None):
		super(EigenvalueCallback, self).__init__()
		self.i = 0
		self.to_dir = to_dir
		self.model_name = model_name
		self.experiment_recorder = experiment_recorder
		self.compression_mode = None
		self.values = {}
		self.tableau20 = [(0, 70, 105), (0, 95, 135), (255, 127, 14), (255, 187, 120),
					 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
					 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
					 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
					 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
		self.patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

		# Process colors
		for i in range(len(self.tableau20)):
			r, g, b = self.tableau20[i]
			self.tableau20[i] = (r / 255., g / 255., b / 255.)

	def gather_vals(self, det, det_corr, eigvals, weight, spectral_norm, layer_name, neuron_id, kernel_id, kernel_dim):
		"""
		Main function to track values

		# Arguments
			:param det: result of np.linalg.det(A)
			:param det_corr: result of np.linalg.det(A.T * A)
			:param eigvals: result of np.linalg.eigvals(A)
			:param weight: (numpy.float) average of the absolute weights in a kernel
			:param spectral_norm: (numpy.float) max singular value of a kernel
			:param layer_name: (string) name of the given layer
			:param neuron_id: (int) ID of the selected neuron
			:param kernel_id: (int) ID of the kernel belonging to the selected neuron
			:param kernel_dim: (int) dim of the kernel (1x1, 3x3, ...)
		"""
		# Handle missing fields
		if layer_name not in self.values:
			self.values[layer_name] = {}
		if neuron_id not in self.values[layer_name]:
			self.values[layer_name][neuron_id] = {}
		if kernel_id not in self.values[layer_name][neuron_id]:
			self.values[layer_name][neuron_id][kernel_id] = {}

		# Store values
		self.values[layer_name][neuron_id][kernel_id]["det"] = det
		self.values[layer_name][neuron_id][kernel_id]["det_corr"] = det_corr
		self.values[layer_name][neuron_id][kernel_id]["eigvals"] = eigvals
		self.values[layer_name][neuron_id][kernel_id]["weight"] = weight
		self.values[layer_name][neuron_id][kernel_id]["spectral_norm"] = spectral_norm
		self.values[layer_name][neuron_id][kernel_id]["kernel_dim"] = kernel_dim

	def is_pruned(self, val_dict, threshold, compression_mode):
		result = False

		# Check for mode: det
		if compression_mode == "det":
			result = np.absolute(val_dict["det"]) < threshold

		# Check for mode: det_corr
		elif compression_mode == "det_corr":
			result = np.absolute(val_dict["det_corr"]) < threshold

		# Check for mode: min_eig
		elif compression_mode == "min_eig":
			eigvals = np.absolute(val_dict["eigvals"])
			result = np.min(eigvals) < threshold

		# Check for mode: min_eig_real
		elif compression_mode == "min_eig_real":
			eigvals = np.absolute(np.real(val_dict["eigvals"]))
			result = np.min(eigvals) < threshold

		# Check for mode: spectral_radius
		elif compression_mode == "spectral_radius":
			eigvals = np.absolute(val_dict["eigvals"])
			result = np.max(eigvals) < threshold

		# Check for mode: spectral_radius_real
		elif compression_mode == "spectral_radius_real":
			eigvals = np.absolute(np.real(val_dict["eigvals"]))
			result = np.max(eigvals) < threshold

		# Check for mode: spectral_norm
		elif compression_mode == "spectral_norm":
			result = val_dict["spectral_norm"] < threshold

		# Check for mode: weight
		elif compression_mode == "weight":
			result = val_dict["weight"] < threshold

		return result

	def plot_eigenvalue_stats(self, threshold, compression_mode, to_file="MicroResNet_eig_stats.png"):
		"""
		Plots the ratio of real/complex eigenvalues used for pruning

		In general, the eigenvalues of a real 3x3 kernel can be:
			(i)   3 distinct real numbers
			(ii)  3 real numbers with repetitions
			(iii) 1 real number and 2 conjugate complex numbers
		"""
		layer_names = []
		total_real_list = []
		pruned_real_list = []
		total_complex_list = []
		pruned_complex_list = []
		for layer in self.values:
			total_real = 0
			pruned_real = 0
			total_complex = 0
			pruned_complex = 0
			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					eigvals = self.values[layer][neuron][kernel]["eigvals"]
					is_complex = np.iscomplex(eigvals)

					if compression_mode == "min_eig":
						temp = np.absolute(eigvals)
						min_eig = np.min(temp)
						is_pruned = min_eig < threshold
						i = np.where(temp == min_eig)[0][0]
						if is_complex[i]:
							total_complex += 1
							if is_pruned:
								pruned_complex += 1
						else:
							total_real += 1
							if is_pruned:
								pruned_real += 1

					elif compression_mode == "min_eig_real":
						temp = np.absolute(np.real(eigvals))
						min_eig_real = np.min(temp)
						is_pruned = min_eig_real < threshold
						i = np.where(temp == min_eig_real)[0][0]
						if is_complex[i]:
							total_complex += 1
							if is_pruned:
								pruned_complex += 1
						else:
							total_real += 1
							if is_pruned:
								pruned_real += 1

					elif compression_mode == "spectral_radius":
						temp = np.absolute(eigvals)
						spectral_radius = np.max(temp)
						is_pruned = spectral_radius < threshold
						i = np.where(temp == spectral_radius)[0][0]
						if is_complex[i]:
							total_complex += 1
							if is_pruned:
								pruned_complex += 1
						else:
							total_real += 1
							if is_pruned:
								pruned_real += 1

					elif compression_mode == "spectral_radius_real":
						temp = np.absolute(np.real(eigvals))
						spectral_radius_real = np.max(temp)
						is_pruned = spectral_radius_real < threshold
						i = np.where(temp == spectral_radius_real)[0][0]
						if is_complex[i]:
							total_complex += 1
							if is_pruned:
								pruned_complex += 1
						else:
							total_real += 1
							if is_pruned:
								pruned_real += 1

			layer_names.append(layer)
			total_real_list.append(total_real)
			pruned_real_list.append(pruned_real)
			total_complex_list.append(total_complex)
			pruned_complex_list.append(pruned_complex)

		# From raw value to percentage
		totals = [i + j for i,j in zip(total_real_list, total_complex_list)]
		pruned_real = [(i / j) * 100 for i,j in zip(pruned_real_list, totals)]
		active_real = [((i - j) / k) * 100 for i,j,k in zip(total_real_list, pruned_real_list, totals)]
		pruned_complex = [(i / j) * 100 for i, j in zip(pruned_complex_list, totals)]
		active_complex = [((i - j) / k) * 100 for i, j, k in zip(total_complex_list, pruned_complex_list, totals)]

		# Draw the chart
		bar_width = 0.35
		opacity = 0.8
		fig = plt.figure(figsize=(23, 11))
		ax1 = fig.add_subplot(1, 1, 1)
		y_pos = np.arange(len(layer_names))
		ax1.bar(y_pos, active_real, bar_width, alpha=opacity, color=self.tableau20[1], label='Active Real', edgecolor=self.tableau20[0], hatch="//")
		ax1.bar(y_pos, pruned_real, bar_width, alpha=opacity, color="#f9bc86", label='Pruned Real', edgecolor="darkorange", hatch="//", bottom=active_real)
		ax1.bar(y_pos, active_complex, bar_width, alpha=opacity, color=self.tableau20[5], label='Active Complex', edgecolor=self.tableau20[4], hatch="--", bottom=[i+j for i,j in zip(active_real, pruned_real)])
		ax1.bar(y_pos, pruned_complex, bar_width, alpha=opacity, color=self.tableau20[6], label='Pruned Complex', edgecolor="darkred", hatch="--", bottom=[i+j+k for i,j,k in zip(active_real, pruned_real, active_complex)])

		ax1.set_xticks(y_pos)
		ax1.set_xticklabels(layer_names, rotation=33)
		ax1.set_ylabel('Percentage (%)')
		ax1.legend(loc="upper left")
		ax1.tick_params(axis="x", labelsize=9)

		# Save the figure
		plt.tight_layout()
		plt.savefig(to_file, bbox_inches="tight")
		plt.close(fig)

		# Log the values
		if self.experiment_recorder is not None:
			vals = {"layer_names": layer_names, "active_real": active_real, "pruned_real": pruned_real, "active_complex": active_complex, "pruned_complex": pruned_complex}
			self.experiment_recorder.record(vals, mode="eig_analysis")

	def count_complex_nums(self, num_array):
		counter_complex = 0
		counter_real = 0
		for num in num_array:
			if np.iscomplex(num):
				counter_complex += 1
			else:
				counter_real += 1
		return counter_complex, counter_real

	def record_eig_stats(self, threshold):
		"""
		Stores the ratio of real/complex eigenvalues used for pruning for each related compression mode

		In general, the eigenvalues of a real 3x3 kernel can be:
			(i)   3 distinct real numbers
			(ii)  3 real numbers with repetitions
			(iii) 1 real number and 2 conjugate complex numbers
		"""
		compression_modes = ["min_eig", "min_eig_real", "spectral_radius", "spectral_radius_real"]
		layer_names = []
		total_real_list = []
		total_complex_list = []
		target_real_list = []
		pruned_real_list = []
		target_complex_list = []
		pruned_complex_list = []
		for layer in self.values:
			total_real = 0
			total_complex = 0
			target_real = {c: 0 for c in compression_modes}
			pruned_real = {c: 0 for c in compression_modes}
			target_complex = {c: 0 for c in compression_modes}
			pruned_complex = {c: 0 for c in compression_modes}

			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					eigvals = self.values[layer][neuron][kernel]["eigvals"]
					num_complex, num_real = self.count_complex_nums(eigvals)
					total_complex += num_complex
					total_real += num_real

					is_complex = np.iscomplex(eigvals)
					for compression_mode in compression_modes:
						if compression_mode == "min_eig":
							temp = np.absolute(eigvals)
							min_eig = np.min(temp)
							is_pruned = min_eig < threshold
							i = np.where(temp == min_eig)[0][0]

						elif compression_mode == "min_eig_real":
							temp = np.absolute(np.real(eigvals))
							min_eig_real = np.min(temp)
							is_pruned = min_eig_real < threshold
							i = np.where(temp == min_eig_real)[0][0]

						elif compression_mode == "spectral_radius":
							temp = np.absolute(eigvals)
							spectral_radius = np.max(temp)
							is_pruned = spectral_radius < threshold
							i = np.where(temp == spectral_radius)[0][0]

						elif compression_mode == "spectral_radius_real":
							temp = np.absolute(np.real(eigvals))
							spectral_radius_real = np.max(temp)
							is_pruned = spectral_radius_real < threshold
							i = np.where(temp == spectral_radius_real)[0][0]

						if is_complex[i]:
							target_complex[compression_mode] += 1
							if is_pruned:
								pruned_complex[compression_mode] += 1
						else:
							target_real[compression_mode] += 1
							if is_pruned:
								pruned_real[compression_mode] += 1

			# Info independent from the selected compression mode
			layer_names.append(layer)
			total_real_list.append(total_real)
			total_complex_list.append(total_complex)

			# Compression mode info
			target_complex_list.append(target_complex)
			pruned_complex_list.append(pruned_complex)
			target_real_list.append(target_real)
			pruned_real_list.append(pruned_real)

		# Log the values
		vals = {
			"layer_names": layer_names,
			"total_complex_list": total_complex_list,
			"total_real_list": total_real_list,
			"target_complex_list": target_complex_list,
			"pruned_complex_list": pruned_complex_list,
			"target_real_list": target_real_list,
			"pruned_real_list": pruned_real_list
		}
		self.experiment_recorder.record(vals, mode="eig_stats")

	def plot_pruning_venn3_diagram(self, threshold, to_file="MicroResNet_venn3.png"):
		"""
		Plots the # of kernels pruned by each kernel pruning criterion

		Venn diagram areas:
			- "a": min_eig
			- "b": weight
			- "c": spectral_radius
		"""
		venn_vals = {"a": 0, "b": 0, "ab": 0, "c": 0, "ac": 0, "bc": 0, "abc": 0}

		for layer in self.values:
			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					val_dict = self.values[layer][neuron][kernel]

					# Check decisions of the different criterias
					min_eig_flag = self.is_pruned(val_dict, threshold, "min_eig")
					weight_flag = self.is_pruned(val_dict, threshold, "weight")
					spectral_radius_flag = self.is_pruned(val_dict, threshold, "spectral_radius")

					# Process Venn diagram data
					key = ""
					if min_eig_flag:
						key += "a"
					if weight_flag:
						key += "b"
					if spectral_radius_flag:
						key += "c"

					venn_vals[key] = venn_vals[key] + 1 if key in venn_vals else 1

		# Prepare the chart data
		groups = ("a", "b", "ab", "c", "ac", "bc", "abc")
		subsets = tuple([venn_vals[g] for g in groups])
		fig = plt.figure(figsize=(11,11))

		# Plot the pie chart
		venn3(subsets=subsets, set_labels=("min_eig", "weight", "spectral_radius"))
		venn3_circles(subsets=subsets, linestyle="dashed", linewidth=1, color="grey")

		# Save the diagram
		plt.savefig(to_file, bbox_inches="tight")
		plt.close(fig)

	def plot_pruning_venn2_diagram(self, threshold, compression_mode, to_file="MicroResNet_venn2.png"):
		"""
		Plots the # of kernels pruned by each kernel pruning criterion

		Venn diagram areas:
			- "a": min_eig 		| spectral_radius
			- "b": min_eig_real | spectral_radius_real
		"""
		venn_vals = {"a": 0, "b": 0, "ab": 0}
		base_mode = "spectral_radius" if "spectral_radius" in compression_mode else "min_eig"
		real_mode = "spectral_radius_real" if "spectral_radius" in compression_mode else "min_eig_real"

		for layer in self.values:
			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					val_dict = self.values[layer][neuron][kernel]

					# Check decisions of the different criterias
					base_flag = self.is_pruned(val_dict, threshold, base_mode)
					real_flag = self.is_pruned(val_dict, threshold, real_mode)

					# Process Venn diagram data
					key = ""
					if base_flag:
						key += "a"
					if real_flag:
						key += "b"
					venn_vals[key] = venn_vals[key] + 1 if key in venn_vals else 1

		# Prepare the chart data
		fig = plt.figure(figsize=(9,9))

		# Plot the pie chart
		venn2(subsets=(venn_vals["a"], venn_vals["b"], venn_vals["ab"]), set_labels=(base_mode, real_mode))
		venn2_circles(subsets=(venn_vals["a"], venn_vals["b"], venn_vals["ab"]), linestyle="dashed", linewidth=1, color="grey")

		# Save the diagram
		plt.savefig(to_file, bbox_inches="tight")
		plt.close(fig)

	def plot_pruning_pie_chart(self, threshold, to_file="MicroResNet_pie.png"):
		"""
		Plots the # of kernels pruned by each kernel pruning criterion

		Pie chart areas:
			- "a": min_eig
			- "b": weight
			- "c": det
			- "d": spectral_radius
			- "e": spectral_norm
			- "f": det_corr
		"""
		pie_chart_vals = {}
		for layer in self.values:
			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					val_dict = self.values[layer][neuron][kernel]
					n = val_dict["kernel_dim"]

					# Check decisions of the different criterias
					min_eig_flag = self.is_pruned(val_dict, threshold, "min_eig")
					weight_flag = self.is_pruned(val_dict, threshold, "weight")
					det_flag = self.is_pruned(val_dict, pow(threshold, n), "det")
					spectral_radius_flag = self.is_pruned(val_dict, threshold, "spectral_radius")
					spectral_norm_flag = self.is_pruned(val_dict, threshold, "spectral_norm")
					det_corr_flag = self.is_pruned(val_dict, pow(threshold, 2*n), "det_corr")

					# Process Venn diagram data
					key = ""
					if min_eig_flag:
						key += "a"
					if weight_flag:
						key += "b"
					if det_flag:
						key += "c"
					if spectral_radius_flag:
						key += "d"
					if spectral_norm_flag:
						key += "e"
					if det_corr_flag:
						key += "f"

					pie_chart_vals[key] = pie_chart_vals[key] + 1 if key in pie_chart_vals else 1

		# Prepare the chart data
		groups = sorted(list(pie_chart_vals.keys()))
		sizes = [pie_chart_vals[g] for g in groups]
		total = sum(sizes)
		sizes = [(x / float(total)) * 100 for x in sizes]
		groups = ["{%s}" % (",".join(list(g))) for g in groups]
		fig = plt.figure(figsize=(11,11))

		# Plot the pie chart
		patches, texts, _ = plt.pie(sizes, labels=groups, autopct="%1.1f%%")
		plt.legend(patches, groups, loc="best")
		plt.axis('equal')
		plt.tight_layout()

		# Save the diagram
		plt.savefig(to_file, bbox_inches="tight")
		plt.close(fig)

	def auto_label_pruning(self, rects, vals):
		i = 0
		for rect in rects:
			height = rect.get_height()
			val = "%.2f%%" % vals[i]
			x_coord = rect.get_x() + rect.get_width() / 2.
			plt.text(x_coord, height + 0.1, val, ha="center", va="bottom", fontsize=12)
			i += 1

	def plot_pruning_bar_chart(self, threshold, to_file="MicroResNet_bar.png"):
		"""
		Plots the # of kernels pruned by each kernel pruning criterion

		Pie chart areas:
			- "a": min_eig
			- "b": weight
			- "c": det
			- "d": spectral_radius
			- "e": spectral_norm
			- "f": det_corr
		"""
		compression_mode_codes = {"a": "min_eig",
								  "b": "weight",
								  "c": "det",
								  "d": "spectral_radius",
								  "e": "spectral_norm",
								  "f": "det_corr"}
		bar_chart_vals = {}

		for layer in self.values:
			for neuron in self.values[layer]:
				for kernel in self.values[layer][neuron]:
					val_dict = self.values[layer][neuron][kernel]
					n = val_dict["kernel_dim"]

					# Check decisions of the different criterias
					min_eig_flag = self.is_pruned(val_dict, threshold, "min_eig")
					weight_flag = self.is_pruned(val_dict, threshold, "weight")
					det_flag = self.is_pruned(val_dict, pow(threshold, n), "det")
					spectral_radius_flag = self.is_pruned(val_dict, threshold, "spectral_radius")
					spectral_norm_flag = self.is_pruned(val_dict, threshold, "spectral_norm")
					det_corr_flag = self.is_pruned(val_dict, pow(threshold, 2*n), "det_corr")

					# Process Venn diagram data
					key = ""
					if min_eig_flag:
						key += "a"
					if weight_flag:
						key += "b"
					if det_flag:
						key += "c"
					if spectral_radius_flag:
						key += "d"
					if spectral_norm_flag:
						key += "e"
					if det_corr_flag:
						key += "f"

					bar_chart_vals[key] = bar_chart_vals[key] + 1 if key in bar_chart_vals else 1

		# Prepare the chart data
		groups = sorted(list(bar_chart_vals.keys()))
		sizes = [bar_chart_vals[g] for g in groups]
		total = sum(sizes)
		sizes = [(x / float(total)) * 100 for x in sizes]
		groups = ["{%s}" % (",".join(list(g))) for g in groups]

		# Plot the bar chart
		bar_width = 0.35
		opacity = 0.8
		fig = plt.figure(figsize=(13, 6))
		ax1 = fig.add_subplot(1, 1, 1)
		y_pos = np.arange(len(sizes))
		rects = ax1.bar(y_pos, sizes, bar_width, alpha=opacity, color=self.tableau20[:len(sizes)])
		self.auto_label_pruning(rects, sizes)
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

		# Log the values
		if self.experiment_recorder is not None:
			vals = {"group_sizes": sizes, "groups": groups, "codes": compression_mode_codes}
			self.experiment_recorder.record(vals, mode="set_analysis")

	def plot_pruning_vs_thresholds(self, to_file="MicroResNet_thresholds.png"):
		"""
		Plot a line chart to show the % of kernels (would be) pruned for different thresholds
		"""
		thresholds = [float('{:.0e}'.format(5 * (x % 2 + 1) * (10 ** (x // 2 - 13)))) for x in range(1, 26)]
		final_vals = {"det": [], "det_corr": [], "min_eig": [], "min_eig_real": [], "spectral_radius": [], "spectral_radius_real": [], "spectral_norm": [], "weight": []}
		for threshold in thresholds:
			counters = {"det": 0, "det_corr": 0, "min_eig": 0, "min_eig_real": 0, "spectral_radius": 0, "spectral_radius_real": 0, "spectral_norm": 0, "weight": 0}
			num_kernels = 0
			for layer in self.values:
				for neuron in self.values[layer]:
					for kernel in self.values[layer][neuron]:
						val_dict = self.values[layer][neuron][kernel]
						num_kernels += 1

						# Check for mode: det
						if self.is_pruned(val_dict, threshold, "det"):
							counters["det"] += 1

						# Check for mode: det
						if self.is_pruned(val_dict, threshold, "det_corr"):
							counters["det_corr"] += 1

						# Check for mode: min_eig
						if self.is_pruned(val_dict, threshold, "min_eig"):
							counters["min_eig"] += 1

						# Check for mode: min_eig_real
						if self.is_pruned(val_dict, threshold, "min_eig_real"):
							counters["min_eig_real"] += 1

						# Check for mode: spectral_radius
						if self.is_pruned(val_dict, threshold, "spectral_radius"):
							counters["spectral_radius"] += 1

						# Check for mode: spectral_radius_real
						if self.is_pruned(val_dict, threshold, "spectral_radius_real"):
							counters["spectral_radius_real"] += 1

						# Check for mode: spectral_norm
						if self.is_pruned(val_dict, threshold, "spectral_norm"):
							counters["spectral_norm"] += 1

						# Check for mode: weight
						if self.is_pruned(val_dict, threshold, "weight"):
							counters["weight"] += 1
			for key in final_vals:
				final_vals[key].append(counters[key] * 100 / num_kernels)

		# Draw the chart
		indices = np.arange(len(thresholds))
		fig = plt.figure(figsize=(18, 9))
		plt.plot(indices, final_vals["det"], color="forestgreen", label="det", linewidth=2)
		plt.plot(indices, final_vals["det_corr"], color="magenta", label="det_corr", linewidth=2)
		plt.plot(indices, final_vals["min_eig"], color="darkolivegreen", label="min_eig", linewidth=2)
		plt.plot(indices, final_vals["min_eig_real"], color="limegreen", label="min_eig_real", linewidth=2)
		plt.plot(indices, final_vals["spectral_radius"], color="red", label="spectral_radius", linewidth=2)
		plt.plot(indices, final_vals["spectral_radius_real"], color="crimson", label="spectral_radius", linewidth=2)
		plt.plot(indices, final_vals["spectral_norm"], color="darkorange", label="spectral_norm", linewidth=2)
		plt.plot(indices, final_vals["weight"], color="#005f87", label="weight", linewidth=2)
		plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0), useOffset=False)
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

		# Log the values
		if self.experiment_recorder is not None:
			vals = {"final_vals": final_vals, "thresholds": thresholds}
			self.experiment_recorder.record(vals, mode="pruning_per_threshold")

	def run_analysis(self,
					 enable_plot_eigenvalue_stats=True,
					 enable_plot_pruning_pie_chart=True,
					 enable_plot_pruning_bar_chart=True,
					 enable_plot_pruning_venn2_diagram=True,
					 enable_plot_pruning_venn3_diagram=False,
					 enable_plot_pruning_vs_thresholds=True):
		# Get the necessary info from a MicroConv2D layer
		threshold = 0
		self.compression_mode = None
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				threshold = layer.get_threshold()
				self.compression_mode = layer.get_compression_mode()
				break

		if enable_plot_eigenvalue_stats and \
				(self.compression_mode == "min_eig" or
				 self.compression_mode == "min_eig_real" or
				 self.compression_mode == "spectral_radius" or
				 self.compression_mode == "spectral_radius_real"):
			self.plot_eigenvalue_stats(threshold, self.compression_mode, to_file="%s%s_eig_stats.png" % (self.to_dir, self.model_name))

		if self.experiment_recorder is not None:
			self.record_eig_stats(threshold)

		if enable_plot_pruning_pie_chart:
			self.plot_pruning_pie_chart(threshold, to_file="%s%s_pie.png" % (self.to_dir, self.model_name))

		if enable_plot_pruning_bar_chart:
			self.plot_pruning_bar_chart(threshold, to_file="%s%s_bar.png" % (self.to_dir, self.model_name))

		if enable_plot_pruning_venn2_diagram and \
				(self.compression_mode == "min_eig" or
				 self.compression_mode == "min_eig_real" or
				 self.compression_mode == "spectral_radius" or
				 self.compression_mode == "spectral_radius_real"):
			self.plot_pruning_venn2_diagram(threshold, self.compression_mode, to_file="%s%s_venn2.png" % (self.to_dir, self.model_name))

		if enable_plot_pruning_venn3_diagram:
			self.plot_pruning_venn3_diagram(threshold, to_file="%s%s_venn3.png" % (self.to_dir, self.model_name))

		if enable_plot_pruning_vs_thresholds:
			self.plot_pruning_vs_thresholds(to_file="%s%s_thresholds.png" % (self.to_dir, self.model_name))

	def run(self):
		"""
		Logic of the callback
		"""
		# Separate storage for each MicroConv2D layer
		for layer in self.model.layers:
			if isinstance(layer, MicroConv2D):
				weights = layer.get_weights()
				shape = weights[0].shape

				# For each neuron
				for neuron in range(shape[3]):
					# For each kernel
					for depth in range(shape[2]):
						# Get the 3x3 convolution matrix
						kernel = weights[0][:, :, depth, neuron]

						# Compute the raw determinant and eigenvalues, and the average of absolute weights
						det = np.linalg.det(kernel)
						det_corr = np.linalg.det(np.dot(kernel.T, kernel))
						eigvals = np.linalg.eigvals(kernel)
						weight = np.average(np.absolute(kernel))
						spectral_norm = np.linalg.norm(kernel, 2)

						# Append to the previous values
						self.gather_vals(det, det_corr, eigvals, weight, spectral_norm, layer.name, neuron, depth, kernel.shape[0])

	def on_train_end(self, logs=None):
		# Gather the final values and export the history
		self.run()
		self.run_analysis()