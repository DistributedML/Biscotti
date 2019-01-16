import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

input_file_directory = "bis_5v_50p_2"
output_file_directory = input_file_directory + "_parsed/"

total_nodes = 50

def parse_logs():

	for i in range(total_nodes):

		fname = input_file_directory + "/log_" + str(i) + "_" + str(total_nodes) + ".log"
		lines = [line.rstrip('\n') for line in open(fname)]

		outfile = open(output_file_directory + "data" + str(i), "w")
		iteration = 0

		for line in lines:

			idx = line.find("Train Error")

			if idx != -1:

				ar_idx = line.find("Attack Rate")

				timestamp = line[7:20]

				outfile.write(str(iteration))
				outfile.write(",")
				outfile.write(line[(idx + 15):(idx + 22)])
				outfile.write(",")
				outfile.write(line[(ar_idx + 15):(ar_idx + 21)])
				outfile.write(",")
				outfile.write(timestamp)
				outfile.write("\n")

				iteration = iteration + 1

		outfile.close()


if __name__ == '__main__':
	
	parse_logs()
