import pdb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

total_nodes = 100
numIterations = 102

colors = ['black', 'red', 'green', 'blue', 'yellow']
labels = ["Federated Learning", 'Biscotti']


def parse_logs(input_file_directory):

	output_file_directory = input_file_directory + "_parsedResults/"


	fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
	lines = [line.rstrip('\n') for line in open(fname)]

	if not os.path.exists(output_file_directory):
		os.makedirs(output_file_directory)

	outfile = open(output_file_directory + "data" +".csv", "w")
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
		
		thisLine =  mlines.Line2D(np.arange(numIterations), dataPoints, color=colors[lineIdx],	linewidth=3, linestyle='-', label=labels[lineIdx])	
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


if __name__ == '__main__':

	foldersToParse = ["Fed_Poison_100_30_mnist", "Bis_Poison_100_30_mnist"]

	parsedFolders = []
	
	for folder in foldersToParse:
		parse_logs(folder)
		parsedFolders.append(folder+"_parsedResults/data.csv")

	plotResults("posion_mnist_30_100.jpg", parsedFolders)