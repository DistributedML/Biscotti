import pdb
import pandas as pd
import os
import numpy as np


# input_file_directory = "FedSys_Azure"
# output_file_directory = input_file_directory + "_parsedResults/"

total_nodes = 100

numRuns=1

def parse_logs():	

	acceptedUpdates = 0
	rejectedUpdates = 0

	for i in range(0,numRuns):

		for j in range(0,total_nodes):
			# pass
			# print("here	")

			fname = "LogFiles_" + str(i) + "/log_"+str(j)+"_" + str(total_nodes)  + ".log"
			print(fname)
			lines = [line.rstrip('\n') for line in open(fname)]

			for line in lines:

				# print(here)
				idx = line.find("Accepting update!")

				if idx != -1:
					acceptedUpdates = acceptedUpdates + 1

				idx = line.find("Rejecting update!")

				if idx != -1:

					rejectedUpdates = rejectedUpdates + 1

	print(acceptedUpdates)
	print(rejectedUpdates)




if __name__ == '__main__':
	
	parse_logs()
