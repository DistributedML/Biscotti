import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

total_nodes = 50

def plot(percent):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((3, 100))

	###########################################
	# across_runs = np.zeros((total_nodes, 102))
	# for i in range(total_nodes):
	# 	print i
	# 	df = pd.read_csv("bis_3v_" + str(percent) + "p_parsed/data" + str(i), header=None)
	# 	across_runs[i] = df[1].values

	# toplot[0] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		print i
		df = pd.read_csv("bis_3v_50p", header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((total_nodes, 102))
	for i in range(total_nodes):
		print i
		df = pd.read_csv("fed_50p", header=None)
		across_runs[i] = df[1].values

	toplot[2] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	# l1 = mlines.Line2D(np.arange(100), toplot[0], 
	# 	color='orange', linestyle='--', linewidth=4, label="3 verifiers")

	l2 = mlines.Line2D(np.arange(100), toplot[1], 
		color='purple', linestyle=':', linewidth=4, label="Biscotti")


	# l2 = mlines.Line2D(np.arange(100), toplot[1], 
	# 	color='purple', linestyle=':', linewidth=4, label="Biscotti")
	
	l3 = mlines.Line2D(np.arange(100), toplot[2], 
		color='black', linestyle='-', linewidth=4, label="Federated learning")

	# ax.add_line(l1)
	ax.add_line(l2)
	ax.add_line(l3)

	# lines=[l1,l2,l3]

	lines = [l3,l2]

	plt.legend(handles=lines, loc='best', fontsize=18)

	plt.xlabel("Iterations", fontsize=22)
	plt.ylabel("Validation Error", fontsize=22)

	axes = plt.gca()
	axes.set_xlim([0, 101])
	axes.set_ylim([0, 1])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)
	# fig.savefig("eval_poisoning" + str(percent) + ".pdf")
	fig.savefig("eval_poisoning.pdf")
	plt.show()

if __name__ == '__main__':
	
	plot(50)
