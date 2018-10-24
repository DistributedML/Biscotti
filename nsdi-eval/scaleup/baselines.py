import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

input_file_directory = "fed_baseline_100"
output_file_directory = input_file_directory + "_parsed/"

total_nodes = 100

def parse_logs():

	for i in range(total_nodes):

		fname = input_file_directory + "/log_" + str(i) + "_" + str(total_nodes) + ".log"
		lines = [line.rstrip('\n') for line in open(fname)]

		outfile = open(output_file_directory + "data" + str(i), "w")
		iteration = 0

		for line in lines:

			idx = line.find("Train Error")

			if idx != -1:

				timestamp = line[7:20]

				outfile.write(str(iteration))
				outfile.write(",")
				outfile.write(line[(idx + 15):(idx + 22)])
				outfile.write(",")
				outfile.write(timestamp)
				outfile.write("\n")

				iteration = iteration + 1

		outfile.close()

def plot(time=True):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 102))

	###########################################
	across_runs = np.zeros((1, 102))
	df = pd.read_csv("fed_baseline_100_parsed/data0", header=None)
	across_runs[0] = df[1].values

	toplot[0] = np.mean(across_runs, axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((100, 102))
	for i in range(total_nodes):
		df = pd.read_csv("parsed_100_1/full_run_" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs, axis=0)
	###########################################

	if time:

		l1 = mlines.Line2D(346 * np.arange(102) / 100, toplot[0], color='black', 
			linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

		l2 = mlines.Line2D(3011 * np.arange(102) / 100, toplot[1], color='red', 
			linewidth=3, linestyle='--', label="Biscotti 100 nodes")
	
	else:

		l1 = mlines.Line2D(np.arange(102), toplot[0], color='black', 
			linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

		l2 = mlines.Line2D(np.arange(102), toplot[1], color='red', 
			linewidth=3, linestyle='--', label="Biscotti 100 nodes")

	ax.add_line(l1)
	ax.add_line(l2)

	plt.legend(handles=[l2, l1], loc='right', fontsize=18)

	axes = plt.gca()
	
	axes.set_ylim([0, 1])

	if time:
		plt.xlabel("Time (s)", fontsize=22)
		axes.set_xlim([0, 3050])
	else:
		plt.xlabel("Training Iterations", fontsize=22)
		axes.set_xlim([0, 100])

	plt.ylabel("Validation Error", fontsize=22)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)
	
	if time:
		fig.savefig("eval_convrate_time.pdf")
	else:
		fig.savefig("eval_convrate.pdf")

	
	# plt.show()


if __name__ == '__main__':
	
	plot(False)
	plot(True)
