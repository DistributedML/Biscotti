import pdb
import pandas as pd
import os
import numpy as np


# input_file_directory = "FedSys_Azure"
# output_file_directory = input_file_directory + "_parsedResults/"

# total_nodes = 100

# numRuns=1

def parse_logs():	

	acceptedUpdates = 0
	rejectedUpdates = 0

	fname = "pingValues"
	# print(fname)
	lines = [line.rstrip('\n') for line in open(fname)]

	minLatency=2000
	maxLatency=0

	totalCount=0
	totalLatency=0
	avgLatency=0

	for line in lines:

		thisLat = float(line)
		totalCount= totalCount + 1
		totalLatency= totalLatency + thisLat

		if thisLat < minLatency:
			minLatency=thisLat

		if thisLat > maxLatency:
			maxLatency=thisLat

	avgLatency=totalLatency/totalCount
	print(avgLatency)
	print(minLatency)
	print(maxLatency)



if __name__ == '__main__':
	
	parse_logs()
