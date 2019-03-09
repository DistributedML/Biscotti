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
numIterations = 102

colors = ['orange', 'red', 'green', 'blue', 'yellow', 'black']
linestyle = ['-', '-', '-', '-', '-', '--']
labels = [r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.5$', r'$\epsilon=1$', r'$\epsilon=2$', 'Biscotti']

def plotResults(outputFile, inputFiles):

	print outputFile
	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((len(inputFiles), numIterations))

	# unpoisonedFedlearn = "results_FedLearn.csv"

	lines = []

	fileIdx = 0
	for inputFile in inputFiles:

		print inputFile
		df =  pd.read_csv(inputFile, header=None)	
		toplot[fileIdx] = df[1].values
		fileIdx+=1

	lineIdx = 0

	print(toplot)
	print(toplot.shape)

	for dataPoints in toplot:
		
		thisLine =  mlines.Line2D(np.arange(numIterations), dataPoints, color=colors[lineIdx],	linewidth=3, linestyle=linestyle[lineIdx], label=labels[lineIdx])	
		ax.add_line(thisLine)
		lines.append(thisLine)
		lineIdx+=1

	# # for line in lines:



	# # ###########################################
	
	# # unpoisonedDF = pd.read_csv(unpoisonedFedlearn, header=None)
	# # toplot[0] = 100 - unpoisonedDF[1].values
	# # l1 = mlines.Line2D(np.arange(200), toplot[0], color='black', 
	# # 	linewidth=3, linestyle='-', label="Federated Learning")	

	# # ###########################################

	# ax.add_line(l1)
	plt.legend(handles=lines, loc='best', fontsize=18)
	
	axes = plt.gca()	

	plt.ylabel("Validation Error", fontsize=22)
	axes.set_ylim([0, 1.0])

	plt.xlabel("Training Iterations", fontsize=22)
	axes.set_xlim([0, numIterations])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)

	fig.savefig(outputFile)

def parse_logs(fname):

	# output_file_directory = input_file_directory + "_parsedResults/"

	# for i in range(0,numRuns):

		# fname = input_file_directory + "/LogFiles_" + str(i) + "/log_0_" + str(total_nodes)  + ".log"
	lines = [line.rstrip('\n') for line in open(fname)]

	outfile = open("parsed_" + filename, "w")
	
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

def plot(firstInput,secondInput,time=True):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, numIterations))

	fedSysOutput = firstInput + "_parsedResults/"
	distSysOutput = secondInput + "_parsedResults/"

	###########################################
	across_runs = np.zeros((numRuns, numIterations))
	for i in range(0,numRuns):
		df = pd.read_csv((fedSysOutput+"data" + str(i)+".csv"), header=None)
		across_runs[i] = df[1].values

	toplot[0] = np.mean(across_runs, axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((numRuns, numIterations))
	for i in range(0,numRuns):
		df = pd.read_csv((distSysOutput+"data" + str(i)+".csv"), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs, axis=0)
	###########################################

	# ###########################################
	# across_runs = np.zeros((numRuns, numIterations))
	# for i in range(0,numRuns):
	# 	df = pd.read_csv("parsed_100_1/full_run_" + str(i), header=None)
	# 	across_runs[i] = df[1].values

	# toplot[1] = np.mean(across_runs, axis=0)
	# ###########################################

	l1 = mlines.Line2D(np.arange(numIterations), toplot[0], color='black', 
		linewidth=3, linestyle='-', label=firstInput)

	l2 = mlines.Line2D(np.arange(numIterations), toplot[1], color='red', 
		linewidth=3, linestyle='--', label=secondInput)

	ax.add_line(l1)
	ax.add_line(l2)

	plt.legend(handles=[l2, l1], loc='right', fontsize=18)

	axes = plt.gca()
	
	axes.set_ylim([0, 0.7])

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
		fig.savefig("privacy-utility-tradeoff.jpg")

	
	# plt.show()


if __name__ == '__main__':

	epsilonValues = ["001", "01", "05","1", "2", "0"]

	parsedFileNames=[]

	for epsilon in epsilonValues:	

		filename = "epsilon_" + epsilon + ".log"
		
		if epsilon == "0":
			
			parsedFileNames.append("parsed_"+filename)
			continue

		else:
			parse_logs(filename)
			parsedFileNames.append("parsed_"+filename)


	plotResults("fig_privacy_utility_krum.pdf", parsedFileNames)
