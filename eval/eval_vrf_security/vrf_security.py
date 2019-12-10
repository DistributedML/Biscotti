import matplotlib
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# for a value of stake try out all stake and see where it is less than zero

line_colors = ['blue', 'orange', 'black']
line_labels = ['Prob of collusion <= 0.001', 'Prob of collusion <= 0.01', 'Prob of collusion <= 0.05']

def main(): 

	# stake_values = [0.1, 0.2, 0.3, 0.4, 0.5]

	# committee_sizes = np.arange(100)

	stake_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

	num_clients = 100

	prob_thresholds = [0.001, 0.01, 0.05]   

	committee_sizes = binomialWithoutReplacement(num_clients, stake_values, prob_thresholds)

	## Add other probability distributions
	
	print(committee_sizes)

	plotStakeVsCommitteeSize(committee_sizes, stake_values, prob_thresholds)

# Uses the formula sum(kCi * (adversary_stake)^i * (adversary_stake)^(N-i))

# Binomial probability modelling two out of three heads. Doesn't take into account distribution of sybils.

def binomialWithoutReplacement(num_clients, stake_values, prob_thresholds):

	committee_sizes_by_prob = []
	
	for prob_threshold in prob_thresholds:

		committee_sizes = np.array(range(3, num_clients))

		committee_values = []

		for stake_value in stake_values:

			for committee_size in committee_sizes:

				startSize = int(committee_size/2) + 1
				# print(startSize)

				majority_idxs = np.array(range(startSize, committee_size+1))

				# print(majority_idxs)

				committee_probability = np.sum([comb(committee_size, idx, exact=True) * (stake_value)**idx * (1-stake_value)**(committee_size - idx) for idx in majority_idxs]) 

				if committee_probability < prob_threshold:
					committee_values.append(committee_size)
					break

		committee_sizes_by_prob.append(committee_values)        
	
	return committee_sizes_by_prob

def plotStakeVsCommitteeSize(committee_sizes, stake_values, prob_thresholds):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 102))

	lines = []

	line_idx = 0

	stake_values = [stake_value * 100 for stake_value in stake_values]

	for prob_threshold in prob_thresholds:

		print(committee_sizes[line_idx])
		line = mlines.Line2D(stake_values[:len(committee_sizes[line_idx])], committee_sizes[line_idx], color=line_colors[line_idx], linewidth=3, linestyle='-', label=line_labels[line_idx])
		lines.append(line)
		line_idx = line_idx + 1

	for line in lines:
		ax.add_line(line)   

	plt.legend(handles=lines, loc='best', fontsize=18, title="Probability of successful collusion")

	axes = plt.gca()    

	plt.xlabel("Adversarial Stake (s)(%)", fontsize=22)

	axes.set_xlim([0, 36])

	plt.ylabel("Committee Size (k)", fontsize=22)

	axes.set_ylim([1, 100])

	plt.yticks(np.arange(0, 100, 5))
	plt.xticks(np.arange(0, 36, 5))

	fig.tight_layout(pad=0.1)
	
	fig.savefig("eval_vrf_security.pdf")

if __name__ == '__main__':
	main()