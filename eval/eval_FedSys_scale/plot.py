import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
from datetime import datetime, timedelta

total_nodes = 200

def parse_logs(input_file_directory, output_file_directory):

	for i in range(num_runs):

		fname = input_file_directory + "/" + str(i) + "/log_0" + "_" + str(total_nodes) + ".log"
		lines = [line.rstrip('\n') for line in open(fname)]

		outfile = open(output_file_directory + "/data" + str(i), "w")
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

# Deprecated function
def plot_uniform(fed_parsed,biscotti_parsed,num_runs,time=True):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 102))

	###########################################
	across_runs = np.zeros((num_runs, 102))
	for i in range(num_runs):
		df = pd.read_csv(fed_parsed + "/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[0] = np.mean(across_runs, axis=0)
	###########################################

	###########################################
	across_runs = np.zeros((num_runs, 102))

	for i in range(num_runs):
		df = pd.read_csv(biscotti_parsed + "/data" + str(i), header=None)
		across_runs[i] = df[1].values

	toplot[1] = np.mean(across_runs, axis=0)
	###########################################

	print toplot[0]
	print toplot[1]


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

	plt.ylabel("Test Error", fontsize=22)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)
	
	if time:
		fig.savefig("eval_convrate_time.pdf")
	else:
		fig.savefig("eval_convrate.pdf")


def plot(fedSysOutput, distSysNoShuffleOutput, numRuns,  time=True):

    fig, ax = plt.subplots(figsize=(10, 5))
    toplot = np.zeros((2, 102))

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    completionTimes = np.zeros(numRuns)
    for i in range(0, numRuns):
        df = pd.read_csv(fed_parsed + "/data" + str(i), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        completionTimes[i] = timeToComplete.seconds

    avgFedSysCompletionTime = np.mean(completionTimes, axis=0)

    toplot[0] = np.mean(across_runs, axis=0)
    print("Fedsys: ")
    print(toplot[0])
    print(completionTimes)
    print(avgFedSysCompletionTime)
    ###########################################
    across_runs = np.zeros((numRuns, 102))
    completionTimes = np.zeros(numRuns)
    for i in range(0, numRuns):
        df = pd.read_csv((distSysNoShuffleOutput + '/data' + str(i)), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        completionTimes[i] = timeToComplete.seconds

    avgDistSysCompletionTime = np.mean(completionTimes, axis=0)

    toplot[1] = np.mean(across_runs, axis=0)
    print("DistSys: ")
    print(toplot[1])
    print(avgDistSysCompletionTime)
    ##########################################

    if time:

        l1 = mlines.Line2D(avgFedSysCompletionTime * np.arange(102) / 100, toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning " + str(total_nodes) + " nodes")

        l2 = mlines.Line2D(avgDistSysCompletionTime * np.arange(102) / 100, toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Biscotti " + str(total_nodes) + " nodes")

    else:

        l1 = mlines.Line2D(np.arange(102), toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning " + str(total_nodes) + " nodes")

        l2 = mlines.Line2D(np.arange(102), toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Biscotti " + str(total_nodes) + " nodes")

    ax.add_line(l1)
    ax.add_line(l2)

    plt.legend(handles=[l1, l2], loc='right', fontsize=18)

    axes = plt.gca()

    axes.set_ylim([0, 1])

    if time:
        plt.xlabel("Time (s)", fontsize=22)
        axes.set_xlim([0, 1000* (int(avgDistSysCompletionTime/1000)+1)])
    else:
        plt.xlabel("Training Iterations", fontsize=22)
        axes.set_xlim([0, 100])

    plt.ylabel("Test Error", fontsize=22)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    fig.tight_layout(pad=0.1)

    if time:
        fig.savefig("eval_convrate_time.pdf")
    else:
        fig.savefig("eval_convrate.pdf")


if __name__ == '__main__':

	if len(sys.argv) != 6:

		print("Usage: [biscotti_input, biscotti_parsed, fed_input, fed_parsed, num_runs]")
		sys.exit()

	biscotti_input = sys.argv[1]
	biscotti_parsed = sys.argv[2]
	fed_input = sys.argv[3]
	fed_parsed = sys.argv[4]
	num_runs = int(sys.argv[5])

	parse_logs(fed_input, fed_parsed)
	parse_logs(biscotti_input, biscotti_parsed)
	
	plot(fed_parsed,biscotti_parsed,num_runs, False)
	plot(fed_parsed,biscotti_parsed,num_runs,True)
