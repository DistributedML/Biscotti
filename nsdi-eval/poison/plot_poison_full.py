import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

input_file_directory = "poison_fed_20p_ar"
output_file_directory = input_file_directory + "_parsed/"

total_nodes = 50

def plot20():

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((5, 100))

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_20p_3v_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[0] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_20p_5v_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_20p_3v_noep_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[2] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_20p_5v_noep_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[3] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		print i
		df = pd.read_csv("poison_fed_20p_ar_parsed/data" + str(i), header=None)
		across_runs[i] = df[2].values

	toplot[4] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	l1 = mlines.Line2D(np.arange(100), toplot[0], 
		color='red', linestyle='-', linewidth=3, label="3 verifiers")

	l2 = mlines.Line2D(np.arange(100), toplot[1], 
		color='green', linestyle='-', linewidth=3, label="5 verifiers")
	
	l3 = mlines.Line2D(np.arange(100), toplot[2], 
		color='red', linestyle='--', linewidth=3, label="3 verifiers, no noise")

	l4 = mlines.Line2D(np.arange(100), toplot[3], 
		color='green', linestyle='--', linewidth=3, label="5 verifiers, no noise")
	
	l5 = mlines.Line2D(np.arange(100), toplot[4], 
		color='orange', linestyle='-', linewidth=3, label="Federated learning")

	ax.add_line(l1)
	ax.add_line(l2)
	ax.add_line(l3)
	ax.add_line(l4)
	ax.add_line(l5)

	plt.legend(handles=[l1, l2, l3, l4, l5], loc='best', fontsize=18)

	plt.xlabel("Iterations", fontsize=22)
	plt.ylabel("Average Training Error", fontsize=22)

	axes = plt.gca()
	axes.set_xlim([0, 101])
	axes.set_ylim([0, 1])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)
	fig.savefig("eval_poisoning20.pdf")
	plt.show()


def plot40():

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((5, 100))

	###########################################
	# across_runs = np.zeros((total_nodes, 102))
	# for i in range(total_nodes):
	# 	df = pd.read_csv("poison_bis_40p_3v_parsed/data" + str(i), header=None)
	# 	across_runs[i] = df[1].values

	# toplot[0] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_full_40p_5v_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	# across_runs = np.zeros((total_nodes, 102))
	# for i in range(total_nodes):
	# 	print i
	# 	df = pd.read_csv("poison_bis_40p_3v_noep_parsed/data" + str(i), header=None)
	# 	across_runs[i] = df[1].values

	# toplot[2] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_bis_full_40p_5v_noep_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[3] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		df = pd.read_csv("poison_fed_40p_full_parsed/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[4] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	# l1 = mlines.Line2D(np.arange(100), toplot[0], 
	# 	color='red', linestyle='-', linewidth=3, label="3 verifiers")

	l2 = mlines.Line2D(np.arange(100), toplot[1], 
		color='green', linestyle='-', linewidth=3, label="5 verifiers")
	
	# l3 = mlines.Line2D(np.arange(100), toplot[2], 
	# 	color='red', linestyle='--', linewidth=3, label="3 verifiers, no noise")

	l4 = mlines.Line2D(np.arange(100), toplot[3], 
		color='green', linestyle='--', linewidth=3, label="5 verifiers, no noise")
	
	l5 = mlines.Line2D(np.arange(100), toplot[4], 
		color='orange', linestyle='-', linewidth=3, label="Federated learning")

	#ax.add_line(l1)
	ax.add_line(l2)
	#ax.add_line(l3)
	ax.add_line(l4)
	ax.add_line(l5)

	plt.legend(handles=[l2, l4, l5], loc='best', fontsize=18)

	plt.xlabel("Iterations", fontsize=22)
	plt.ylabel("Average Training Error", fontsize=22)

	axes = plt.gca()
	axes.set_xlim([0, 101])
	axes.set_ylim([0, 1.1])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)
	fig.savefig("eval_poisoning40.pdf")
	plt.show()


if __name__ == '__main__':
	
	plot40()
