import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import os

total_nodes = 100

colors = ['blue', 'orange', 'green']
linestyle = ['-', '--', ':']

def parse_logs(input_log_dir, output_file_dir,fname, directory):

	fname = input_log_dir+"/"+fname
	lines = [line.rstrip('\n') for line in open(fname)]

	outfile = open(output_file_dir + "/" + directory + "_parsed.csv", "w")
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

def plot(output_file_dir, dirs):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((len(dirs), 102))

	i = 0
	l = []

	for directory in dirs:

		###########################################		
		df = pd.read_csv(output_file_dir + "/" + directory+"_parsed.csv" , header=None)
		toplot[i] = df[1].values
		
		###########################################

		line = mlines.Line2D(np.arange(102), toplot[i], 
		color=colors[i], linestyle=linestyle[i], linewidth=4, label=directory + " nodes per minute")
		l.append(line)
		ax.add_line(line)
		i=i+1


	plt.legend(handles=l, loc='best', fontsize=18)

	plt.xlabel("Iterations", fontsize=18)
	plt.ylabel("Test Error", fontsize=18)

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

	input_file_dir = sys.argv[1]
	output_file_dir = sys.argv[2]
	fname=sys.argv[3]

	dirs = [x for x in os.listdir(input_file_dir)]

	for directory in dirs:
		input_log_dir = input_file_dir + "/" + directory
		print input_log_dir
		parse_logs(input_log_dir, output_file_dir, fname, directory)

	plot(output_file_dir,dirs)
