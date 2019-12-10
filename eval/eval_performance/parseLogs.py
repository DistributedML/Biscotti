import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import csv

import locale
locale.setlocale(locale.LC_ALL, '')

# if len(sys.argv) != 2:
#     print(
#         "Usage: [100, biscotti_output_file_dir, biscotti_input_file_dir, fedsys_output_file_dir, fedsys_input_file_dir]")
#     sys.exit()
# Example Usage: python generateResults.py 100 biscottiParsedResults/ BiscottiLogs fedSysParsedResults/ FedSysLogs

# total_nodes = sys.argv[1]
total_nodes = 0


# fedsys_output_file_dir = sys.argv[4]
# fedsys_input_file_dir = sys.argv[5]

def parse_logs(numRuns, input_file_directory, output_file_directory):
    
    for i in range(0, numRuns):

        fname = input_file_directory + "/" + str(i) + "/log_0_" + str(total_nodes) + ".log"
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
        parse_noise(input_file_directory + "/" + str(i), output_file_directory, i)


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
        parse_verif(input_file_directory + "/" +  str(i), output_file_directory, i)


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
        parse_aggr(input_file_directory  + "/" + str(i), output_file_directory, i)


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


def getAvgTotalTime(parsed_files_directory,num_runs):
    completionTimes = np.zeros(num_runs)
    for i in range(0, num_runs):
        df = pd.read_csv((parsed_files_directory + 'data' + str(i)), header=None)
        startTime = datetime.strptime(df[2].values[0], "%H:%M:%S.%f")
        endTime = datetime.strptime(df[2].values[101], "%H:%M:%S.%f")
        if endTime < startTime:
            endTime += timedelta(days=1)
        timeToComplete = endTime - startTime
        completionTimes[i] = timeToComplete.seconds

    totalAvg = np.mean(completionTimes, axis=0)
    dev = np.std(completionTimes/100)
    return totalAvg, dev

def getAvg(parsed_files_directory,num_runs):

    completionTime = np.zeros(num_runs)
    allTimes = []
    for i in range(0, num_runs):
        df = pd.read_csv((parsed_files_directory + 'data' + str(i)), header=None)
        # print(df)
        # print(completionTime)
        # print(df[1].values)
        allTimes.append(df[1].values)
        completionTime[i] = np.sum(df[1].values)
        print(completionTime[i])

    print(completionTime/100)

    totalAvg = np.mean(completionTime)
    dev = np.std(completionTime/100)
    print(dev)
    return totalAvg, dev

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: [input log directory, parsed results directory, num_runs in input directory]")
        sys.exit()

    input_file_dir = sys.argv[1]
    output_file_dir = sys.argv[2]
    num_runs = int(sys.argv[3])

    print(input_file_dir)

    dirs = [x for x in os.listdir(input_file_dir)]
    
    rows = []

    for directory in dirs:
        
        input_log_dir = input_file_dir + directory 
        output_log_dir = output_file_dir +  directory + "parsedLogs/"
        output_agg_dir = output_file_dir +  directory + "parsedAggr/"
        output_verif_dir = output_file_dir +  directory + "parsedVerif/"
        output_noising_dir = output_file_dir +  directory + "parsedNoising/"

        total_nodes = directory

        print(input_log_dir)

        parse_logs(num_runs, input_log_dir, output_log_dir)
        parse_all_aggr(input_log_dir, output_agg_dir, num_runs)
        parse_all_verif(input_log_dir, output_verif_dir, num_runs)
        parse_all_noise(input_log_dir, output_noising_dir, num_runs)    

        print(directory)
        aggrAvg, aggrDev = getAvg(output_agg_dir,num_runs)
        print(aggrDev)
        aggrAvg, aggrDev = aggrAvg/100, aggrDev
        verifAvg, verifDev = getAvg(output_verif_dir,num_runs) 
        verifAvg, verifDev = verifAvg/100, verifDev
        noisingAvg, noisDev = getAvg(output_noising_dir,num_runs)
        noisingAvg, noisDev = noisingAvg/100, noisDev
        totalTime, totalDev = getAvgTotalTime(output_log_dir,num_runs) 
        totalTime, totalDev = totalTime/100, totalDev
        floodingTime = totalTime - aggrAvg - verifAvg - noisingAvg 
        floodingDev = totalDev - aggrDev - verifDev - noisDev 

        print("For " + str(directory) + " nodes:")
        print("Avg Aggr:" + str(aggrAvg))
        print("Avg Verif:" + str(verifAvg))
        print("Avg Noising: " + str(noisingAvg))
        print("Avg Flooding: " + str(floodingTime))
        print("Avg total: " + str(totalTime))

        row = [total_nodes,noisingAvg,noisDev,verifAvg,verifDev,aggrAvg,aggrDev,floodingTime,floodingDev,totalTime,totalDev]

        rows.append(row)

        # break

    columns = ['num_nodes', 'noising', 'noisingDev','verification', 'verificationDev','sec_agg','sec_aggDev' ,'flooding', 'floodingDev', 'total', 'totalDev']

    print rows

    results_df = pd.DataFrame(rows,columns=columns)
    results_df.to_csv('incremental_results.csv',index=False)    

