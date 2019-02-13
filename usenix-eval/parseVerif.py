import os
import sys
from datetime import datetime, timedelta

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

def get_completion_time(startTime, endTime):
    startTime = datetime.strptime(startTime, "%H:%M:%S.%f")
    endTime = datetime.strptime(endTime, "%H:%M:%S.%f")
    if endTime < startTime:
        endTime += timedelta(days=1)
    completionTime = endTime - startTime
    return str(completionTime)

def get_highest_id(list):
    max = -1
    for number in list:
        if str(number) > max:
            max = number
    return max

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
                    completionTime
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

def parse_aggr(input_file_directory, output_file_directory, i):
    fname = input_file_directory + "/log_0_" + str(total_nodes) + ".log"
    lines = [line.rstrip('\n') for line in open(fname)]

    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory)

    outfile = open(output_file_directory + "data" + str(i), "w")
    print(outfile)

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


if __name__ == '__main__':
    # parse_logs(3, "./eval-performance/LogsKrumNoShuffle/", "./eval-performance/parsedNoShuffle/")
    # parse_logs(3, "./eval-performance/LogsKrumShuffle/", "./eval-performance/parsedShuffle/")
    #parse_verif("./performance-breakdown/1", "./performance-breakdown/parsedVerif/", 1)
    parse_aggr("./performance-breakdown/1", "./performance-breakdown/parsedAggregation/", 1)
