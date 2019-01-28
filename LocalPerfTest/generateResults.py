import pdb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
from datetime import datetime, timedelta

if len(sys.argv) != 6:
    print(
        "Usage: [100, biscotti_output_file_dir, biscotti_input_file_dir, fedsys_output_file_dir, fedsys_input_file_dir]")
    sys.exit()
# Example Usage: python generateResults.py 100 biscottiParsedResults/ BiscottiLogs fedSysParsedResults/ FedSysLogs

total_nodes = sys.argv[1]
biscotti_output_file_dir = sys.argv[2]
biscotti_input_file_dir = sys.argv[3]
fedsys_output_file_dir = sys.argv[4]
fedsys_input_file_dir = sys.argv[5]


def parse_logs(numRuns, input_file_directory, output_file_directory):
    for i in range(0, numRuns):

        fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
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


def plot(numRuns, fedSysInput, distSysInput, fedSysOutput, distSysOutput, time=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    toplot = np.zeros((2, 102))

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    maxFedSysCompletionTime = timedelta(seconds=0)
    for i in range(0, numRuns):
        df = pd.read_csv((fedSysOutput + 'data' + str(i)), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        if timeToComplete > maxFedSysCompletionTime:
            maxFedSysCompletionTime = timeToComplete

    toplot[0] = np.mean(across_runs, axis=0)
    maxFedSysCompletionTime = maxFedSysCompletionTime.total_seconds()

    ###########################################

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    maxDistSysCompletionTime = timedelta(seconds=0)
    for i in range(0, numRuns):
        df = pd.read_csv((distSysOutput + 'data' + str(i)), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        if timeToComplete > maxDistSysCompletionTime:
            maxDistSysCompletionTime = timeToComplete

    toplot[1] = np.mean(across_runs, axis=0)
    maxDistSysCompletionTime = maxDistSysCompletionTime.total_seconds()
    ###########################################

    # ###########################################
    # across_runs = np.zeros((numRuns, 102))
    # for i in range(0,numRuns):
    # 	df = pd.read_csv("parsed_100_1/full_run_" + str(i), header=None)
    # 	across_runs[i] = df[1].values

    # toplot[1] = np.mean(across_runs, axis=0)
    # ###########################################

    if time:

        # fedSysTime =
        # distSysTime =

        l1 = mlines.Line2D(maxFedSysCompletionTime * np.arange(102) / 100, toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

        l2 = mlines.Line2D(maxDistSysCompletionTime * np.arange(102) / 100, toplot[1], color='red',
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
        axes.set_xlim([0, 14000])
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
    parse_logs(3, biscotti_input_file_dir, biscotti_output_file_dir)
    parse_logs(3, fedsys_input_file_dir, fedsys_output_file_dir)

    #plot(3,"FedSys_Azure","azure-deploy",False)
    plot(3, fedsys_input_file_dir, biscotti_input_file_dir, fedsys_output_file_dir, biscotti_output_file_dir, True)
