import pdb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# input_file_directory = "FedSys_Azure"
# output_file_directory = input_file_directory + "_parsedResults/"

# input_file_directory_Biscotti = "azure_deploy"
# output_file_directory = input_file_directory + "_parsedResults/"

total_nodes = 100


def parse_logs(numRuns,input_file_directory):

	output_file_directory = input_file_directory + "_parsedResults/"

	for i in range(0,numRuns):

		fname = input_file_directory + "/LogFiles_" + str(i) + "/log_0_" + str(total_nodes)  + ".log"
		lines = [line.rstrip('\n') for line in open(fname)]

		if not os.path.exists(output_file_directory):
			os.makedirs(output_file_directory)

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

def plot(numRuns,fedSysInput,distSysInput,time=True):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 102))

	fedSysOutput = fedSysInput + "_parsedResults/"
	distSysOutput = distSysInput + "_parsedResults/"

	###########################################
	across_runs = np.zeros((numRuns, 102))
	for i in range(0,numRuns):
		df = pd.read_csv((fedSysOutput+'data'+str(i)), header=None)
		across_runs[i] = df[1].values

	toplot[0] = np.mean(across_runs, axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((numRuns, 102))
	for i in range(0,numRuns):
		df = pd.read_csv((distSysOutput+'data'+str(i)), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs, axis=0)
	###########################################

	# ###########################################
	# across_runs = np.zeros((numRuns, 102))
	# for i in range(0,numRuns):
	# 	df = pd.read_csv("parsed_100_1/full_run_" + str(i), header=None)
	# 	across_runs[i] = df[1].values

	# toplot[1] = np.mean(across_runs, axis=0)
	# ###########################################

	if time:

		l1 = mlines.Line2D(300 * np.arange(102) / 100, toplot[0], color='black', 
			linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

		l2 = mlines.Line2D(7110 * np.arange(102) / 100, toplot[1], color='red', 
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
		axes.set_xlim([0, 8000])
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
		fig.savefig("eval_convrate_time.jpg")
	else:
		fig.savefig("eval_convrate.jpg")

	
	# plt.show()


if __name__ == '__main__':
	
	# parse_logs(3, "FedSys_Azure")

	# plot(3,"FedSys_Azure","azure-deploy",False)
	plot(3,"FedSys_Azure","azure-deploy",True)
