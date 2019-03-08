import pdb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
from datetime import datetime, timedelta

if len(sys.argv) != 2:
    print(
        "Usage: [100, biscotti_output_file_dir, biscotti_input_file_dir, fedsys_output_file_dir, fedsys_input_file_dir]")
    sys.exit()
# Example Usage: python generateResults.py 100 biscottiParsedResults/ BiscottiLogs fedSysParsedResults/ FedSysLogs

total_nodes = sys.argv[1]
#biscotti_output_file_dir = sys.argv[2]
#biscotti_input_file_dir = sys.argv[3]
#fedsys_output_file_dir = sys.argv[4]
#fedsys_input_file_dir = sys.argv[5]


def parse_logs(numRuns, input_file_directory, output_file_directory):
    for i in range(1, numRuns):

        fname = input_file_directory + str(i) + "/log_0_" + str(total_nodes) + ".log"
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


def plot(numRuns, fedSysOutput, distSysShuffleOutput, distSysNoShuffleOutput, time=True):
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
    print("Fedsys: " + str(maxFedSysCompletionTime))

    ###########################################

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    maxDistSysShuffleCompletionTime = timedelta(seconds=0)
    for i in range(0, numRuns):
        df = pd.read_csv((distSysShuffleOutput + 'data' + str(i)), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        if timeToComplete > maxDistSysShuffleCompletionTime:
            maxDistSysShuffleCompletionTime = timeToComplete

    toplot[1] = np.mean(across_runs, axis=0)
    maxDistSysShuffleCompletionTime = maxDistSysShuffleCompletionTime.total_seconds()
    print("Shuffle: " + str(maxDistSysShuffleCompletionTime))
    ###########################################

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    maxDistSysNoShuffleCompletionTime = timedelta(seconds=0)
    for i in range(0, numRuns):
        df = pd.read_csv((distSysNoShuffleOutput + 'data' + str(i)), header=None)
        across_runs[i] = df[1].values
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        if timeToComplete > maxDistSysNoShuffleCompletionTime:
            maxDistSysNoShuffleCompletionTime = timeToComplete

    toplot[1] = np.mean(across_runs, axis=0)
    maxDistSysNoShuffleCompletionTime = maxDistSysNoShuffleCompletionTime.total_seconds()
    print("No Shuffle: " + str(maxDistSysNoShuffleCompletionTime))
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

        l2 = mlines.Line2D(maxDistSysShuffleCompletionTime * np.arange(102) / 100, toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Krum with Shuffle 100 nodes")

        l3 = mlines.Line2D(maxDistSysNoShuffleCompletionTime * np.arange(102) / 100, toplot[1], color='blue',
                           linewidth=3, linestyle='--', label="Krum no Shuffle 100 nodes")

    else:

        l1 = mlines.Line2D(np.arange(102), toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

        l2 = mlines.Line2D(np.arange(102), toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Biscotti 100 nodes")

    ax.add_line(l1)
    ax.add_line(l2)
    ax.add_line(l3)

    plt.legend(handles=[l3, l2, l1], loc='right', fontsize=18)

    axes = plt.gca()

    axes.set_ylim([0, 1])

    if time:
        plt.xlabel("Time (s)", fontsize=22)
        axes.set_xlim([0, 4000])
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
        fig.savefig("optimization_comparisons.jpg")
    else:
        fig.savefig("eval_convrate.jpg")


# plt.show()


def plot2(numRuns, fedSysOutput, distSysNoShuffleOutput, time=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    toplot = np.zeros((2, 102))

    ###########################################
    across_runs = np.zeros((numRuns, 102))
    completionTimes = np.zeros(3)
    for i in range(0, numRuns):
        df = pd.read_csv((fedSysOutput + 'data' + str(i)), header=None)
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
    completionTimes = np.zeros(3)
    for i in range(0, numRuns):
        df = pd.read_csv((distSysNoShuffleOutput + 'data' + str(i)), header=None)
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

        l1 = mlines.Line2D(avgFedSysCompletionTime * np.arange(102) / 100, toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

        l2 = mlines.Line2D(avgDistSysCompletionTime * np.arange(102) / 100, toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Biscotti 100 nodes")

    else:

        l1 = mlines.Line2D(np.arange(102), toplot[0], color='black',
                           linewidth=3, linestyle='-', label="Federated Learning 100 nodes")

        l2 = mlines.Line2D(np.arange(102), toplot[1], color='red',
                           linewidth=3, linestyle='--', label="Biscotti 100 nodes")

    ax.add_line(l1)
    ax.add_line(l2)

    plt.legend(handles=[l1, l2], loc='right', fontsize=18)

    axes = plt.gca()

    axes.set_ylim([0, 1])

    if time:
        plt.xlabel("Time (s)", fontsize=22)
        axes.set_xlim([0, 4000])
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



if __name__ == '__main__':
    #parse_logs(3, "./eval-performance/FedSysLogs/", "./eval-performance/parsedFedSys/")
    #parse_logs(3, "./eval-performance/FedSysLogs/", "./eval-performance/parsedNoShuffle/")
    #parse_logs(3, "./eval-performance/LogsKrumNoShuffle/", "./eval-performance/parsedNoShuffle/")
    #parse_logs(3, "./eval-performance/LogsKrumShuffle/", "./eval-performance/parsedShuffle/")
    #plot(3, "./eval-performance/parsedFedSys/", "./eval-performance/parsedShuffle/", "./eval-performance/parsedNoShuffle/", True)
    #plot(3, "./eval-performance/parsedFedSys/", "./eval-performance/parsedShuffle/", "./eval-performance/parsedNoShuffle/", False)
    plot2(3, "./eval-performance/parsedFedSys/", "./eval-performance/parsedNoShuffle/", True)
    plot2(3, "./eval-performance/parsedFedSys/", "./eval-performance/parsedNoShuffle/", False)