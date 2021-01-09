"""
Title           :run.py
Description     :Benchmark code
Author          :Ilke Cugu
Date Created    :19-02-2019
Date Modified   :13-06-2020
version         :4.7.5
python_version  :3.6.6
"""

import os
import gc
import argparse
import configparser
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from callbacks.EigenvalueCallback import EigenvalueCallback
from callbacks.CompressionCallback import CompressionCallback
from callbacks.ComperativeTestingCallback import ComperativeTestingCallback
from callbacks.RegularizationCallback import RegularizationCallback
from callbacks.HistoryCallback import HistoryCallback
from applications.microresnet.MicroResNet import MicroResNet
from testers.CIFAR10_tester import CIFAR10_tester
from testers.CIFAR100_tester import CIFAR100_tester
from testers.tinyimagenet_tester import tinyimagenet_tester
from analysis.ExperimentRecorder import ExperimentRecorder
from tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def reset_keras(sess_hist, sess_model):
	sess = K.get_session()
	K.clear_session()
	sess.close()
	K.get_session()

	try:
		del sess_hist
		del sess_model
	except:
		pass

	gc.collect()

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = True
	K.set_session(tf.Session(config=sess_config))

compression_modes_str = """
	Supported compression modes for benchmarking:
		- all							: includes every algorithm specified below
		- det							: abs determinant of the kernel
		- det_corr						: abs determinant of the Grammian matrix of the given kernel K (K.T * K)
		- det_contrib					: relative abs determinant of the kernel w.r.t all kernels within a given neuron
		- det_sorted_kernels			: for each neuron bottom X%% of the kernels are killed w.r.t abs determinants
		- det_sorted_neurons			: sum of kernel abs determinants determines the significance and bottom X%% of the neurons are killed
		- min_eig						: min abs eigenvalue of the kernel
		- min_eig_real					: min abs eigenvalue (real parts only) of the kernel
		- min_eig_contrib				: relative min abs eigenvalue of the kernel w.r.t all kernels within a given neuron
		- min_eig_real_contrib			: relative min abs eigenvalue (real parts only) of the kernel w.r.t all kernels within a given neuron
		- min_eig_sorted_kernels		: for each neuron bottom X%% of the kernels are killed w.r.t min abs eigenvalues
		- min_eig_sorted_neurons		: sum of kernel abs min eigenvalues determines the significance and bottom X%% of the neurons are killed
		- spectral_radius				: max abs eigenvalue of the kernel
		- spectral_radius_real			: max abs eigenvalue (real parts only) of the kernel
		- spectral_radius_contrib		: relative spectral radius of the kernel w.r.t all kernels within a given neuron
		- spectral_radius_real_contrib	: relative spectral radius (real parts only) of the kernel w.r.t all kernels within a given neuron
		- spectral_radius_sorted_kernels: for each neuron bottom X%% of the kernels are killed w.r.t spectral radii
		- spectral_radius_sorted_neurons: sum of kernel spectral radii determines the significance and bottom X%% of the neurons are killed
		- spectral_norm					: max singular value of the kernel
		- spectral_norm_contrib			: relative spectral norm of the kernel w.r.t all kernels within a given neuron
		- spectral_norm_sorted_kernels	: for each neuron bottom X%% of the kernels are killed w.r.t spectral norms
		- spectral_norm_sorted_neurons	: sum of kernel spectral norms determines the significance and bottom X%% of the neurons are killed
		- weight						: sum of abs weights of the kernel
		- weight_contrib				: relative sum of abs weights of the kernel w.r.t all kernels within a given neuron
		- weight_sorted_kernels			: for each neuron bottom X%% of the kernels are killed w.r.t sum of abs kernel weights
		- weight_sorted_neurons			: (Li et al. ICLR 2017) sum of abs kernel weights determines the significance and bottom X%% of the neurons are killed
		- random_kernels				: randomly killing kernels
		- random_neurons				: randomly killing neurons
"""

datasets_str = """
	Supported datasets for benchmarking:
		- CIFAR-10
		- CIFAR-100
		- tiny-imagenet
"""

def create_config_file(config):
	# Default configurations
	config["DEFAULT"] = {"version": "4.7.5",
						 "depth": 32,
						 "lr": 1e-3,
						 "batch_size": 128,
						 "epochs": 400,
						 "steps": 2,
						 "history_interval": 0,
						 "optimizer": "adam",
						 "momentum": 0.9,
						 "decay": 1e-4,
						 "l1_penalty": 1e-4,
						 "significance_threshold": 1e-4,
						 "contribution_threshold": 1e-3,
						 "compression_rate": 0.2,
						 "compression_mode": "weight",
						 "loss": "categorical_crossentropy",
						 "datasets": "CIFAR-10",
						 "verbose": 0,
						 "print_compression_stats": False,
						 "print_confusion_matrix": False,
						 "print_model_arch": False,
						 "print_config": True,
						 "run_1st_stage": False,
						 "run_2nd_stage": True,
						 "stop_after_pruning": False,
						 "save_model": False,
						 "no_res_connection": False,
						 "custom_regularization": False}

	with open("settings.ini", "w+") as config_file:
		config.write(config_file)

if __name__ == '__main__':

	# Dynamic parameters
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--depth", help="# of layers", type=int)
	parser.add_argument("--lr", help="learning rate", type=float)
	parser.add_argument("--batch_size", help="batch size", type=int)
	parser.add_argument("--epochs", help="# of epochs", type=int)
	parser.add_argument("--steps", help="(x + 1) steps: there will always be a final stage, so steps = 3 means (2 stage + 1 final stage)", type=int)
	parser.add_argument("--history_interval", help="if set >0, it will enable HistoryCallback to log compression info per defined # of epochs ", type=int)
	parser.add_argument("--optimizer", help="optimization algorithm", type=str)
	parser.add_argument("--momentum", help="momentum (only relevant if the 'optimizer' algorithm is using it)", type=float)
	parser.add_argument("--decay", help="weight decay (only relevant if the 'optimizer' algorithm is using it)", type=float)
	parser.add_argument("--l1_penalties", help="L1 regularizer penalties for benchmarking", type=float, nargs="+")
	parser.add_argument("--significance_threshold", help="compression threshold for (det, det_corr, min_eig, min_eig_real, spectral_radius, spectral_radius_real, spectral_norm, weight) modes", type=float)
	parser.add_argument("--contribution_threshold", help="compression threshold for (det_contrib, min_eig_contrib, min_eig_real_contrib, spectral_radius_contrib, spectral_radius_real_contrib, spectral_norm_contrib, weight_contrib) modes", type=float)
	parser.add_argument("--compression_rate", help="compression rate for controllable compression algorithms", type=float)
	parser.add_argument("--compression_modes", help=compression_modes_str, nargs="+")
	parser.add_argument("--modes_to_compare", help="alternative compression modes to simulate & compare multiple approaches", nargs="+")
	parser.add_argument("--pretrained_weights", help=".h5 file path to load pre-trained weights", type=str)
	parser.add_argument("--loss", help="loss function", type=str)
	parser.add_argument("--datasets", help=datasets_str, nargs="+")
	parser.add_argument("--verbose", help="Keras verbose", type=int)
	parser.add_argument("--print_compression_stats", help="prints the layer-by-layer pruning stats", action="store_true")
	parser.add_argument("--print_confusion_matrix", help="prints the confusion matrix", action="store_true")
	parser.add_argument("--print_model_arch", help="prints the model architecture", action="store_true")
	parser.add_argument("--print_config", help="prints the active configurations", action="store_true")
	parser.add_argument("--run_1st_stage", help="training of the baseline ResNet models", action="store_true")
	parser.add_argument("--run_2nd_stage", help="training of MicroResNet models", action="store_true")
	parser.add_argument("--stop_after_pruning", help="disables retraining of the compressed model", action="store_true")
	parser.add_argument("--save_model", help="to save the trained models", action="store_true")
	parser.add_argument("--to_dir", help="filepath to save charts, models, etc.", type=str)
	parser.add_argument("--no_res_connection", help="to disable the residual connections", action="store_true")
	parser.add_argument("--custom_regularization", help="to enable custom weight regularization callback", action="store_true")
	args = vars(parser.parse_args())

	# Static parameters
	config = configparser.ConfigParser(allow_no_value=True)
	try:
		if not os.path.exists("settings.ini"):
			create_config_file(config)

		# Override the default values if specified
		config.read("settings.ini")
		temp = dict(config["DEFAULT"])
		temp.update({k: v for k, v in args.items() if v is not None})
		config.read_dict({"DEFAULT": temp})
		config = config["DEFAULT"]

		# Assign the active values
		version = config["version"]
		depth = int(config["depth"])
		lr = float(config["lr"])
		batch_size = int(config["batch_size"])
		epochs = int(config["epochs"])
		steps = max(2, int(config["steps"])) # (x + 1) steps: there will always be a final stage, so steps = 3 means (2 stage + 1 final stage)
		history_interval = int(config["history_interval"])
		optimizer = config["optimizer"]
		momentum = float(config["momentum"])
		decay = float(config["decay"])
		l1_penalty = float(config["l1_penalty"])
		significance_threshold = float(config["significance_threshold"])
		contribution_threshold = float(config["contribution_threshold"])
		compression_rate = float(config["compression_rate"])
		compression_mode = config["compression_mode"]
		modes_to_compare = config["modes_to_compare"] if "modes_to_compare" in config else None
		pretrained_weights = config["pretrained_weights"] if "pretrained_weights" in config else None
		loss = config["loss"]
		datasets = config["datasets"]
		verbose = int(config["verbose"])
		PRINT_COMPRESSION_STATS = config.getboolean("print_compression_stats")
		PRINT_CONFUSION_MATRIX = config.getboolean("print_confusion_matrix")
		PRINT_MODEL_ARCH = config.getboolean("print_model_arch")
		PRINT_CONFIG = config.getboolean("print_config")
		RUN_1ST_STAGE = config.getboolean("run_1st_stage")
		RUN_2ND_STAGE = config.getboolean("run_2nd_stage")
		STOP_AFTER_PRUNING = config.getboolean("stop_after_pruning")
		SAVE_MODEL = config.getboolean("save_model")
		NO_RES_CONNECTION = config.getboolean("no_res_connection")
		CUSTOM_REGULARIZATION = config.getboolean("custom_regularization")
		log("Configuration is completed.")
	except Exception as e:
		log("Error: " + str(e), LogType.ERROR)
		log("Configuration fault! New settings.ini is created. Restart the program.", LogType.ERROR)
		create_config_file(config)
		exit(1)

	# Process benchmark parameters
	log("Model compression experiment...")

	# Process directory path to save files
	if args["to_dir"] is None:
		to_dir = ""
	else:
		to_dir = "%s/" % args["to_dir"]
		if not os.path.isdir(to_dir):
			os.mkdir(to_dir)

	# Process l1 penalty parameters
	if args["l1_penalties"] is not None and len(args["l1_penalties"]) > 0:
		if args["l1_penalties"][0] == -13:
			l1_penalties = [None, 1e-6, 1e-5, 4e-5, 1e-4]
			# For logging the active configurations
			config["l1_penalties"] = str(l1_penalties)
		elif args["l1_penalties"][0] < 0:
			l1_penalties = [None]
			# For logging the active configurations
			config["l1_penalties"] = str(l1_penalties)
		else:
			l1_penalties = args["l1_penalties"]
	else:
		l1_penalties = [l1_penalty]

	# Process compression modes
	if args["compression_modes"] is not None and len(args["compression_modes"]) > 0:
		compression_modes = ["det",
							 "det_corr",
							 "det_contrib",
							 "det_sorted_kernels",
							 "det_sorted_neurons",
							 "min_eig",
							 "min_eig_real",
							 "min_eig_contrib",
							 "min_eig_real_contrib",
							 "min_eig_sorted_kernels",
							 "min_eig_sorted_neurons",
							 "spectral_radius",
							 "spectral_radius_real",
							 "spectral_radius_contrib",
							 "spectral_radius_real_contrib",
							 "spectral_radius_sorted_kernels",
							 "spectral_radius_sorted_neurons",
							 "spectral_norm",
							 "spectral_norm_contrib",
							 "spectral_norm_sorted_kernels",
							 "spectral_norm_sorted_neurons",
							 "weight",
							 "weight_contrib",
							 "weight_sorted_kernels",
							 "weight_sorted_neurons",
							 "random_kernels",
							 "random_neurons"]
		# Mode checker
		for s in args["compression_modes"]:
			if s not in compression_modes and s != "all":
				log("Nice try... but %s is not an allowed compression mode!" % s, LogType.ERROR)
				exit(1)

		# Handle specific mode selections
		if "all" in args["compression_modes"]:
			# For logging the active configurations
			config["compression_modes"] = str(compression_modes)
		else:
			compression_modes = args["compression_modes"]
	else:
		compression_modes = [compression_mode]

	# Process selected alternative compression modes for benchmarking
	if args["modes_to_compare"] is not None and len(args["modes_to_compare"]) > 0:
		modes_to_compare = ["det",
							"det_corr",
							"det_contrib",
							"min_eig",
							"min_eig_real",
							"min_eig_contrib",
							"min_eig_real_contrib",
							"spectral_radius",
							"spectral_radius_real",
							"spectral_radius_contrib",
							"spectral_radius_real_contrib",
							"spectral_norm",
							"spectral_norm_contrib",
							"weight",
							"weight_contrib"]
		# Mode checker
		for s in args["modes_to_compare"]:
			if s not in modes_to_compare and s != "all":
				log("Nice try... but %s is not an allowed compression mode for performance comparison!" % s, LogType.ERROR)
				exit(1)

		# Handle specific mode selections
		if "all" in args["modes_to_compare"]:
			# For logging the active configurations
			config["modes_to_compare"] = str(modes_to_compare)
		else:
			modes_to_compare = args["modes_to_compare"]
	else:
		modes_to_compare = ["det", "det_corr", "min_eig", "min_eig_real", "spectral_radius", "spectral_radius_real", "spectral_norm", "weight"]

	# Process selected datasets for benchmarking
	if args["datasets"] is not None and len(args["datasets"]) > 0:
		datasets = ["CIFAR-100",
					"CIFAR-10",
					"MNIST",
					"tiny-imagenet"]
		# Dataset checker
		for s in args["datasets"]:
			if s not in datasets and s != "all":
				log("Nice try... but %s is not an allowed dataset!" % s, LogType.ERROR)
				exit(1)

		# Handle specific dataset selections
		if "all" in args["datasets"]:
			# For logging the active configurations
			config["datasets"] = str(datasets)
		else:
			datasets = args["datasets"]
	else:
		datasets = [datasets]

	# Log the active configuration if needed
	if PRINT_CONFIG:
		log_config(config)

	# Prepare the benchmarks
	testers = {"CIFAR-10": CIFAR10_tester(wait=True),
			   "CIFAR-100": CIFAR100_tester(wait=True),
			   "tiny-imagenet": tinyimagenet_tester(wait=True)}
	log("Benchmarks are initialized.")

	for dataset in datasets:
		log("%s benchmark is started." % dataset)

		tester = testers[dataset]
		tester.activate() # manually trigger the dataset loader
		n_classes = tester.get_n_classes()
		input_shape = tester.get_input_shape()
		y_test = tester.get_y_test()

		#  -----------------------------------------------------------------------------------------------------------------  #
		# |                                 First Stage: train the base ResNet models                                       | #
		#  -----------------------------------------------------------------------------------------------------------------  #

		if RUN_1ST_STAGE:
			# Build the base ResNet model
			if NO_RES_CONNECTION:
				model_name = "NoResNet%s[%s]" % (depth, dataset)
				model = MicroResNet(input_shape, n_classes, depth, name=model_name, disable_compression=True, disable_residual_connections=True, pretrained_weights=pretrained_weights)
			else:
				model_name = "ResNet%s[%s]" % (depth, dataset)
				model = MicroResNet(input_shape, n_classes, depth, name=model_name, disable_compression=True, pretrained_weights=pretrained_weights)

			log("Baseline model is ready.")

			# Train the baseline model
			log("Baseline model training...")
			callbacks = [EarlyStopping(monitor="val_acc", patience=epochs, verbose=verbose, restore_best_weights=True)]
			hist, score = tester.run(model,
									 optimizer=optimizer,
									 lr=lr,
									 momentum=momentum,
									 decay=decay,
									 loss=loss,
									 batch_size=batch_size,
									 epochs=epochs,
									 verbose=verbose,
									 callbacks=callbacks)

			log("%s Total param: %s" % (model_name, model.count_params()))
			log("%s Test loss: %s" % (model_name, score[0]))
			log("%s Test accuracy: %s" % (model_name, score[1]))
			log("----------------------------------------------------------------")

			# Plot and save the learning curve
			chart_path = "%s_learning_curve.png" % model_name
			plot_learning_curve(hist.history, chart_path)

			# Plot and save the model architecture
			chart_path = "%s_arch.png" % model_name
			model.plot_model(chart_path)

			# Save the baseline model & print its structure
			if SAVE_MODEL:
				model.save("%s.h5" % model_name)
				model.summary()

			del model

		#  -----------------------------------------------------------------------------------------------------------------  #
		# |                     Second Stage: train MicroResNet with dynamic model compression                            | #
		#  -----------------------------------------------------------------------------------------------------------------  #

		if RUN_2ND_STAGE:
			for compression_mode in compression_modes:
				for l1_penalty in l1_penalties:
					# Init
					hist = {
						'val_loss': [],
						'val_acc': [],
						'loss': [],
						'acc': []
					}
					model = None
					threshold = contribution_threshold if "contrib" in compression_mode else significance_threshold
					prefix = "No" if NO_RES_CONNECTION else ""
					model_name = "Micro%sResNet%s[%s][l1=%s][mode=%s][threshold=%s]" % (prefix, depth, dataset, to_scientific(l1_penalty), compression_mode, to_scientific(threshold))
					log("%s training..." % model_name)

					# Create an experiment recorder to accumulate statistical information
					init_mode = "random_init" if pretrained_weights is None else "imagenet_init" if pretrained_weights == "imagenet" else "static_init"
					experiment_recorder = ExperimentRecorder("Micro%sResNet%s" % (prefix, depth),
															 optimizer,
															 loss,
															 lr,
															 batch_size,
															 epochs // steps,
															 init_mode,
															 dataset,
															 l1_penalty,
															 history_interval=history_interval,
															 threshold=threshold,
															 compression_mode=compression_mode,
															 verbose=verbose)

					for i in range(steps):
						# Remove weight regularization from the final step
						active_l1_penalty = None if i == steps - 1 else l1_penalty
						disable_compression = True if i == steps - 1 else False

						# Build the MicroResNet model
						model = MicroResNet(input_shape,
											n_classes,
											depth,
											name=model_name,
											l1_penalty=active_l1_penalty,
											significance_threshold=significance_threshold,
											contribution_threshold=contribution_threshold,
											disable_compression=disable_compression,
											pretrained_MicroResNet=model,
											compression_mode=compression_mode,
											compression_rate=compression_rate,
											pretrained_weights=pretrained_weights)

						# Run the benchmark
						if i == steps - 1:
							if STOP_AFTER_PRUNING:
								break

							# In the last step, there is no compression, so save the best weights for final testing
							callbacks = [EarlyStopping(monitor="val_acc", patience=epochs // steps, verbose=verbose, restore_best_weights=True), CompressionCallback()]

						else:
							# During active compression, log and analyze the pruning criterias
							active_recorder = experiment_recorder if i == 0 else None # For now, only a single compression step is supported
							callbacks = [RegularizationCallback(active_l1_penalty)] if CUSTOM_REGULARIZATION else []

							# Optional history logger for detailed inspection of the model compression
							if history_interval > 0:
								callbacks.append(HistoryCallback(model_name, tester, epochs // steps, history_interval, modes_to_compare, to_dir, active_recorder))

							# Append rest of the callbacks
							callbacks.append(EigenvalueCallback(model_name, to_dir, active_recorder))
							callbacks.append(ComperativeTestingCallback(tester, modes_to_compare, active_recorder))
							callbacks.append(CompressionCallback())

						temp_hist, score = tester.run(model,
													  optimizer=optimizer,
													  lr=lr,
													  momentum=momentum,
													  decay=decay,
													  loss=loss,
													  batch_size=batch_size,
													  epochs=epochs // steps,
													  verbose=verbose,
													  callbacks=callbacks)

						# Cumulative history
						hist['val_loss'].extend(temp_hist.history['val_loss'])
						hist['val_acc'].extend(temp_hist.history['val_acc'])
						hist['loss'].extend(temp_hist.history['loss'])
						hist['acc'].extend(temp_hist.history['acc'])

						# Evaluate the model with the test data
						log("%s Step: %s" % (model_name, i + 1))
						log("%s Total param: %s" % (model_name, model.count_params()))
						log("%s Test loss: %s" % (model_name, score[0]))
						log("%s Test accuracy: %s" % (model_name, score[1]))

					# Check if the pruning is bug-free
					if not model.neural_activity_check():
						log("Model compression failure!!!", LogType.ERROR)

					# Log layer by layer compression stats
					if PRINT_COMPRESSION_STATS:
						log("%s Compression Stats:" % model_name)
						for report in model.report_compression_stats():
							log(report)

					log("----------------------------------------------------------------")

					# Save the MicroResNet model & print its structure
					if SAVE_MODEL:
						model.save("%s%s.h5" % (to_dir, model_name))
						model.summary()

					# Plot and save the learning curve
					chart_path = "%s%s_learning_curve.png" % (to_dir, model_name)
					plot_learning_curve(hist, chart_path, experiment_recorder)

					# Plot and save the compression stats
					chart_path = "%s%s_compression_per_layer.png" % (to_dir, model_name)
					model.plot_compression_per_layer(chart_path)

					# Plot and save the confusion matrix
					if PRINT_CONFUSION_MATRIX:
						y_preds = tester.predict(model)
						chart_path = "%s%s_confusion_matrix.png" % (to_dir, model_name)
						_y_test = one_hot_to_int(y_test)
						_y_preds = one_hot_to_int(y_preds)
						plot_confusion_matrix(_y_test, _y_preds, chart_path, n_classes=n_classes)

					# Plot and save the model architecture
					if PRINT_MODEL_ARCH:
						chart_path = "%s%s_arch.png" % (to_dir, model_name)
						model.plot_model(chart_path)

					# Update the experiment info file
					experiment_recorder.save_data()

					reset_keras(hist, model)

			log("# ------------------------------------------------------ #")
	log("Done.")
