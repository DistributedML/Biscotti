import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import csv

import locale
import re
locale.setlocale(locale.LC_ALL, '')

total_nodes = 0
DEFAULT_VALUE="3"
NODE_NUM = ["3", "5", "10", "26"]

def get_completion_time(startTime, endTime):
    startTime = datetime.strptime(startTime, "%H:%M:%S.%f")
    endTime = datetime.strptime(endTime, "%H:%M:%S.%f")
    if endTime < startTime:
        endTime += timedelta(days=1)
    completionTime = endTime - startTime
    return str(completionTime.seconds)


def parse_logs(runFiles):

    avgIteration = []
    
    for runFile in runFiles:

        lines = [line.rstrip('\n') for line in open(runFile)]

        iteration = 0

        iterationTimes = []

        startTime = lines[0][7:20]

        for line in lines:

            idx = line.find("Train Error")

            if idx != -1:

                endTime = line[7:20]
 
                timeToComplete = get_completion_time(startTime, endTime)               
                startTime=endTime
                iterationTimes.append(int(timeToComplete))

        # print(iterationTimes)

        if len(iterationTimes) == 0:
            continue

        avgIter = np.mean(np.array(iterationTimes))
        avgIteration.append(avgIter)

    avg = np.mean(np.array(avgIteration))
    dev = np.std(np.array(avgIteration))

    return avg,dev



if __name__ == '__main__':

    if len(sys.argv) != 4:

        print("Usage: [input log directory, parsed results directory, num_runs in input directory]")
        sys.exit()

    input_file_dir = sys.argv[1]
    output_file_dir = sys.argv[2]
    num_runs = int(sys.argv[3])

    print(input_file_dir)

    files = [x for x in os.listdir(input_file_dir)]


    cols = [[NODE + "_avg", NODE + "_dev"] for NODE in NODE_NUM]
    cols = [item for sublist in cols for item in sublist]
    columns = ['committee_type'] + cols

    results_df = pd.DataFrame(columns=columns)

    row_list = []

    row = ['noisers']

    for NODES in NODE_NUM:

        str_noiser_regex = NODES + "_" + DEFAULT_VALUE + "_" + DEFAULT_VALUE + "_\\d.log"
        noiser_regex = re.compile(str_noiser_regex)
        different_runs = [input_file_dir+file for file in files if re.match(noiser_regex, file)]

        str_noiser_regex = "\\d+" + "_" + DEFAULT_VALUE + "_" + DEFAULT_VALUE + "_\\d.log"
       
        print(NODES)
        avg, dev = parse_logs(different_runs)
        print(avg)
        print(dev)
        row.append(avg)
        row.append(dev)

    row_list.append(row)
    print(row_list)


    row = ['verifiers']

    # Calculate for verifiers
    for NODES in NODE_NUM:

        str_regex = DEFAULT_VALUE + "_" + NODES + "_" + DEFAULT_VALUE +  "_\\d.log" 
        regex = re.compile(str_regex)
        different_runs = [input_file_dir+file for file in files if re.match(regex, file)]
        print(NODES)
        avg, dev = parse_logs(different_runs)
        print(avg)
        print(dev)
        row.append(avg)
        row.append(dev)

    row_list.append(row)       

    row = ['aggregators']

    # Calculate for aggregators
    for NODES in NODE_NUM:

        str_regex = DEFAULT_VALUE + "_" + DEFAULT_VALUE + "_" + NODES +  "_\\d.log" 
        regex = re.compile(str_regex)
        different_runs = [input_file_dir+file for file in files if re.match(regex, file)]
        print(NODES)
        avg, dev = parse_logs(different_runs)
        print(avg)
        print(dev)
        row.append(avg)
        row.append(dev)

    row_list.append(row)
    print(row_list)

    results_df = pd.DataFrame(row_list,columns=columns)
    results_df.to_csv('eval_vrf.csv',index=False)  

