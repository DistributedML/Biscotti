import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

total_nodes = 100

def parse_logs():

	fname = "60s.log"
	lines = [line.rstrip('\n') for line in open(fname)]

	outfile = open("60s_parsed.csv", "w")
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

	fig, ax = plt.subplots(figsize=(10, 2.5))
	toplot = np.zeros((3, 102))

	###########################################
	df = pd.read_csv("15s_parsed.csv", header=None)
	toplot[0] = df[1].values
	###########################################

	###########################################
	df = pd.read_csv("30s_parsed.csv", header=None)
	toplot[1] = df[1].values
	###########################################

	###########################################
	df = pd.read_csv("60s_parsed.csv", header=None)
	toplot[2] = df[1].values
	###########################################

	l1 = mlines.Line2D(np.arange(102), toplot[0], 
		color='blue', linestyle='-', linewidth=4, label="4 nodes/minute")

	l2 = mlines.Line2D(np.arange(102), toplot[1], 
		color='orange', linestyle='--', linewidth=4, label="2 nodes/minute")

	l3 = mlines.Line2D(np.arange(102), toplot[2], 
		color='green', linestyle=':', linewidth=4, label="1 node/minute")
	
	ax.add_line(l1)
	ax.add_line(l2)
	ax.add_line(l3)

	plt.legend(handles=[l1, l2, l3], loc='best', fontsize=18)

	plt.xlabel("Iterations", fontsize=18)
	plt.ylabel("Validation Error", fontsize=18)

	axes = plt.gca()
	axes.set_xlim([0, 101])
	axes.set_ylim([0, 1])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=20)
	plt.setp(ax.get_yticklabels(), fontsize=20)

	fig.tight_layout(pad=0.1)
	fig.savefig("eval_churn.pdf")
	plt.show()


if __name__ == '__main__':
	
	# parse_logs()
	plot()
