import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    print(
        "Usage: [100, biscotti_output_file_dir, biscotti_input_file_dir, fedsys_output_file_dir, fedsys_input_file_dir]")
    sys.exit()
# Example Usage: python generateResults.py 100 biscottiParsedResults/ BiscottiLogs fedSysParsedResults/ FedSysLogs

total_nodes = sys.argv[1]


# biscotti_output_file_dir = sys.argv[2]
# biscotti_input_file_dir = sys.argv[3]
# fedsys_output_file_dir = sys.argv[4]
# fedsys_input_file_dir = sys.argv[5]

def parse_logs(numRuns, input_file_directory, output_file_directory):
    for i in range(0, numRuns):

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


def get_completion_time(startTime, endTime):
    startTime = datetime.strptime(startTime, "%H:%M:%S.%f")
    endTime = datetime.strptime(endTime, "%H:%M:%S.%f")
    if endTime < startTime:
        endTime += timedelta(days=1)
    completionTime = endTime - startTime
    return str(completionTime.seconds)


def get_highest_id(list):
    max = -1
    for number in list:
        if str(number) > max:
            max = number
    return max


def parse_all_noise(input_file_directory, output_file_directory, numFiles):
    for i in range(0, numFiles):
        parse_noise(input_file_directory + str(i), output_file_directory, i)


def parse_noise(input_file_directory, output_file_directory, i):
    fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
    lines = [line.rstrip('\n') for line in open(fname)]

    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory)

    outfile = open(output_file_directory + "data" + str(i), "w")

    noisingNumber = 0

    for i in range(0, len(lines)):
        line = lines[i]
        idx = line.find("Getting noise from")
        if idx != -1:
            startTime = line[7:20]

            for j in range(i, len(lines)):
                line2 = lines[j]
                if line2.find("Sending update to verifiers") != -1:
                    endTime = line2[7:20]
                    completionTime = get_completion_time(startTime, endTime)
                    outfile.write(str(noisingNumber))
                    outfile.write(",")
                    outfile.write(completionTime)
                    outfile.write("\n")
                    noisingNumber = noisingNumber + 1
                    break
    outfile.close()


def parse_all_verif(input_file_directory, output_file_directory, numFiles):
    for i in range(0, numFiles):
        parse_verif(input_file_directory + str(i), output_file_directory, i)


def parse_verif(input_file_directory, output_file_directory, i):
    fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
    lines = [line.rstrip('\n') for line in open(fname)]

    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory)

    outfile = open(output_file_directory + "data" + str(i), "w")

    verificationNumber = 0

    for i in range(0, len(lines)):
        line = lines[i]
        idx = line.find("Sending update to verifiers")
        if idx != -1:
            startTime = line[7:20]

            for j in range(i, len(lines)):
                line2 = lines[j]
                if line2.find("Couldn't get enough signatures") != -1 or line2.find("Sending update to miners") != -1:
                    endTime = line2[7:20]
                    completionTime = get_completion_time(startTime, endTime)
                    outfile.write(str(verificationNumber))
                    outfile.write(",")
                    outfile.write(completionTime)
                    outfile.write("\n")
                    verificationNumber = verificationNumber + 1
                    break
    outfile.close()


def parse_aggr_for_iteration(input_file_directory, iteration, lead_miner):
    fname = input_file_directory + "/log_" + str(lead_miner) + "_" + str(total_nodes) + ".log"
    lines = [line.rstrip('\n') for line in open(fname)]
    for i in range(0, len(lines)):
        line = lines[i]
        idx = line.find("Got share for " + str(iteration) + ", I am at " + str(iteration))
        if idx != -1:
            startTime = line[7:20]

            for j in range(i, len(lines)):
                line2 = lines[j]
                if line2.find("Sending block of iteration: " + str(iteration)) != -1:
                    endTime = line2[7:20]
                    completionTime = get_completion_time(startTime, endTime)
                    return completionTime


def parse_all_aggr(input_file_directory, output_file_directory, numFiles):
    for i in range(0, numFiles):
        parse_aggr(input_file_directory + str(i), output_file_directory, i)


def parse_aggr(input_file_directory, output_file_directory, i):
    fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
    lines = [line.rstrip('\n') for line in open(fname)]

    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory)

    outfile = open(output_file_directory + "data" + str(i), "w")

    iteration = 0

    for i in range(0, len(lines)):
        line = lines[i]
        idx = line.find("Miners are")

        if idx != -1:
            miners = line[48:len(line) - 1]
            miners = miners.split(" ")
            leadMiner = get_highest_id(miners)
            completionTime = parse_aggr_for_iteration(input_file_directory, iteration, leadMiner)
            outfile.write(str(iteration))
            outfile.write(",")
            outfile.write(str(completionTime))
            outfile.write("\n")
            iteration = iteration + 1

    outfile.close()


def getAvgTotalTime(parsed_files_directory):
    completionTimes = np.zeros(3)
    for i in range(0, 3):
        df = pd.read_csv((parsed_files_directory + 'data' + str(i)), header=None)
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        completionTimes[i] = timeToComplete.seconds

    totalAvg = np.mean(completionTimes, axis=0)
    return totalAvg


def getAvg(parsed_files_directory):
    completionTime = [[], [], []]
    for i in range(0, 3):
        df = pd.read_csv((parsed_files_directory + 'data' + str(i)), header=None)
        completionTime[i] = np.sum(df[1].values)
    totalAvg = np.mean(completionTime)
    return totalAvg

if __name__ == '__main__':
    parse_logs(3, "./performance-breakdown/40Nodes/", "./performance-breakdown/40Nodes/parsedLogs/")
    parse_all_aggr("./performance-breakdown/40Nodes/", "./performance-breakdown/40Nodes/parsedAggr/", 3)
    parse_all_verif("./performance-breakdown/40Nodes/", "./performance-breakdown/40Nodes/parsedVerif/", 3)
    parse_all_noise("./performance-breakdown/40Nodes/", "./performance-breakdown/40Nodes/parsedNoising/", 3)
    aggrAvg100 = getAvg("./performance-breakdown/100Nodes/parsedAggr/") / 100
    verifAvg100 = getAvg("./performance-breakdown/100Nodes/parsedVerif/") / 100
    noisingAvg100 = getAvg("./performance-breakdown/100Nodes/parsedNoising/") / 100
    totalTime100 = getAvgTotalTime("./performance-breakdown/100Nodes/parsedLogs/") / 100
    floodingTime100 = totalTime100 - aggrAvg100 - verifAvg100 - noisingAvg100
    print("Avg Aggr 100 Nodes: " + str(aggrAvg100))
    print("Avg Verif 100 Nodes: " + str(verifAvg100))
    print("Avg Noising 100 Nodes: " + str(noisingAvg100))
    print("Avg Flooding 100 Nodes: " + str(floodingTime100))
    print("Avg total time: " + str(totalTime100))
    print("")
    aggrAvg80 = getAvg("./performance-breakdown/80Nodes/parsedAggr/") / 100
    verifAvg80 = getAvg("./performance-breakdown/80Nodes/parsedVerif/") / 100
    noisingAvg80 = getAvg("./performance-breakdown/80Nodes/parsedNoising/") / 100
    totalTime80 = getAvgTotalTime("./performance-breakdown/80Nodes/parsedLogs/") / 100
    floodingTime80 = totalTime80 - aggrAvg80 - verifAvg80 - noisingAvg80
    print("Avg Aggr 80 Nodes: " + str(aggrAvg80))
    print("Avg Verif 80 Nodes: " + str(verifAvg80))
    print("Avg Noising 80 Nodes: " + str(noisingAvg80))
    print("Avg Flooding 80 Nodes: " + str(floodingTime80))
    print("Avg total time: " + str(totalTime80))
    print("")
    aggrAvg60 = getAvg("./performance-breakdown/60Nodes/parsedAggr/") / 100
    verifAvg60 = getAvg("./performance-breakdown/60Nodes/parsedVerif/") / 100
    noisingAvg60 = getAvg("./performance-breakdown/60Nodes/parsedNoising/") / 100
    totalTime60 = getAvgTotalTime("./performance-breakdown/60Nodes/parsedLogs/") / 100
    floodingTime60 = totalTime60 - aggrAvg60 - verifAvg60 - noisingAvg60
    print("Avg Aggr 60 Nodes: " + str(aggrAvg60))
    print("Avg Verif 60 Nodes: " + str(verifAvg60))
    print("Avg Noising 60 Nodes: " + str(noisingAvg60))
    print("Avg Flooding 60 Nodes: " + str(floodingTime60))
    print("Avg total time: " + str(totalTime60))
    print("")
    aggrAvg40 = getAvg("./performance-breakdown/40Nodes/parsedAggr/") / 100
    verifAvg40 = getAvg("./performance-breakdown/40Nodes/parsedVerif/") / 100
    noisingAvg40 = getAvg("./performance-breakdown/40Nodes/parsedNoising/") / 100
    totalTime40 = getAvgTotalTime("./performance-breakdown/40Nodes/parsedLogs/") / 100
    floodingTime40 = totalTime40 - aggrAvg40 - verifAvg40 - noisingAvg40
    print("Avg Aggr 40 Nodes: " + str(aggrAvg40))
    print("Avg Verif 40 Nodes: " + str(verifAvg40))
    print("Avg Noising 40 Nodes: " + str(noisingAvg40))
    print("Avg Flooding 40 Nodes: " + str(floodingTime40))
    print("Avg total time: " + str(totalTime40))
