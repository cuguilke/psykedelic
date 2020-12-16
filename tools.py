"""
Title           :tools.py
Description     :Generic functions to plot/visualize/represent benchmark results
Author          :Ilke Cugu
Date Created    :07-04-2019
Date Modified   :02-05-2020
version         :1.6.0
python_version  :3.6.6
"""

import logging
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from time import localtime, strftime
from enum import Enum

logging.basicConfig(filename="experiment.log", filemode="a", format="%(message)s", level=logging.INFO)

# Limit unwanted logging messages from packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

class LogType(Enum):
	DEBUG = 0
	INFO = 1
	WARNING = 2
	ERROR = 3

def log(msg, log_type=LogType.INFO, to_file=True, to_stdout=True):
	msg = "%s %s" % (get_time(), msg)

	if to_stdout:
		print(msg)
	if to_file and log_type == LogType.DEBUG:
		logging.debug(msg)
	elif to_file and log_type == LogType.INFO:
		logging.info(msg)
	elif to_file and log_type == LogType.WARNING:
		logging.warning(msg)
	elif to_file and log_type == LogType.ERROR:
		logging.error(msg)

def log_config(config):
	compression_modes_defined = True if "compression_modes" in config else False
	l1_penalties_defined = True if "l1_penalties" in config else False

	log("Active Configuration:")
	log("--------------------")
	for key in config:
		# Do not showthe default benchmark parameters if they are overriden
		if key == "compression_mode" and compression_modes_defined:
			continue
		elif key == "l1_penalty" and l1_penalties_defined:
			continue

		residual = 24 - len(key)
		temp = ""
		while len(temp) < residual:
			temp += " "
		log("%s%s: %s" % (key, temp, config[key]))

def to_scientific(x):
	return "{:.0e}".format(x)

def get_time():
	return "[%s]" % strftime("%a, %d %b %Y %X", localtime())

def one_hot_to_int(preds):
	result = np.empty(preds.shape[0])
	for i in range(preds.shape[0]):
		result[i] = np.argmax(preds[i])
	return result

def plot_learning_curve(training_hist, chart_path, experiment_recorder=None):
	"""
	Plots the learning curve of the given training history

	# Arguments
		:param training_hist: (hist.history) of keras.models.Model.fit
		:param chart_path: (String) file path for the output chart
		:param experiment_recorder: (ExperimentRecorder) for cumulative logging of empirical results
	"""
	is_ok = True

	# Error handler for missing values
	for key in ["acc", "loss", "val_acc", "val_loss"]:
		if key not in training_hist:
			is_ok = False

	if is_ok:
		# Starting building the learning curve graph
		fig, ax1 = plt.subplots(figsize=(14, 9))
		epoch_list = list(range(1, len(training_hist['acc']) + 1))

		# Plotting training and test losses
		train_loss, = ax1.plot(epoch_list, training_hist['loss'], color='red', alpha=.5)
		val_loss, = ax1.plot(epoch_list, training_hist['val_loss'], linewidth=2, color='green')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')

		# Plotting test accuracy
		ax2 = ax1.twinx()
		train_accuracy, = ax2.plot(epoch_list, training_hist['acc'], linewidth=1, color='orange')
		val_accuracy, = ax2.plot(epoch_list, training_hist['val_acc'], linewidth=2, color='blue')
		ax2.set_ylim(bottom=0, top=1)
		ax2.set_ylabel('Accuracy')

		# Adding legend
		plt.legend([train_loss, val_loss, val_accuracy, train_accuracy], ['Training Loss', 'Validation Loss', 'Validation Accuracy', 'Training Accuracy'], loc=7, bbox_to_anchor=(1, 0.8))
		plt.title('Learning Curve')

		# Saving learning curve
		plt.savefig(chart_path)
		plt.close(fig)

		# Log the values
		if experiment_recorder is not None:
			experiment_recorder.record({"hist": training_hist}, mode="learning_curve")

def plot_confusion_matrix(y_test, y_preds, chart_path, n_classes, class_labels=None):
	class_labels = [""]*n_classes if class_labels is None else class_labels

	#Generate the normalized confusion matrix
	cm = confusion_matrix(y_test, y_preds)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig = plt.figure(figsize=(33, 26))
	plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(n_classes)
	plt.xticks(tick_marks, class_labels, rotation=30)
	plt.yticks(tick_marks, class_labels)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], '.1f'),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	# Saving learning curve
	plt.savefig(chart_path)
	plt.close(fig)