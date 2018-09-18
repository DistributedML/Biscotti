import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

input_file_directory = "run_40_1/"
output_file_directory = "parsed_40_1/"

total_nodes = 40

def parse_logs():

	for i in range(total_nodes):

		fname = input_file_directory + "log_" + str(i) + "_" + str(total_nodes) + ".log"
		lines = [line.rstrip('\n') for line in open(fname)]

		outfile = open(output_file_directory + "full_run_" + str(i), "w")
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

def plot():

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 100))

	###########################################
	across_runs = np.zeros((100, 102))
	for i in range(total_nodes):
		df = pd.read_csv("parsed_40_1/full_run_" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[0] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((100, 102))
	for i in range(total_nodes):
		df = pd.read_csv("parsed_100_1/full_run_" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs[:, 0:100], axis=0)
	###########################################

	l1 = mlines.Line2D(np.arange(100), toplot[0], color='black', label="Biscotti 40 nodes")
	l2 = mlines.Line2D(np.arange(100), toplot[1], color='red', label="Biscotti 100 nodes")
	
	ax.add_line(l1)
	ax.add_line(l2)

	plt.legend(handles=[l1, l2], loc='right', fontsize=18)

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
	fig.savefig("eval_convrate.pdf")
	plt.show()


if __name__ == '__main__':
	
	plot()
