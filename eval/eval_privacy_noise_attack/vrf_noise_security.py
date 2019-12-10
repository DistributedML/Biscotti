import matplotlib
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# for a value of stake try out all stake and see where it is less than zero

line_colors = ['blue', 'orange', 'green']
line_labels = ['# Noisers = 3', '# Noisers = 5', '# Noisers = 10']

num_verifiers = 3   

def main(): 

	stake_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

	num_clients = 100
	num_noisers = [3, 5, 10]
	probabilities = binomialWithoutReplacement(stake_values, num_noisers)	

	plotStakeVsProbability(probabilities, stake_values, num_noisers)

	print probabilities


# Binomial probability modelling two out of three heads. Doesn't take into account distribution of sybils.

def binomialWithoutReplacement(stake_values, num_noisers):

	prob_by_num_noisers = []

	for num_noiser in num_noisers:

		this_noiser_prob = []

		for stake_value in stake_values:

			committee_probability_noise = (stake_value)**num_noiser  

			committee_probability_verifier = 1 - (stake_value)**num_verifiers

			prob = committee_probability_noise * committee_probability_verifier

			this_noiser_prob.append(prob) 

		prob_by_num_noisers.append(this_noiser_prob)        
	
	return prob_by_num_noisers

def plotStakeVsProbability(probabilities, stake_values, num_noisers):

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((2, 102))

	lines = []

	line_idx = 0

	# stake_values = [stake_value * 100 for stake_value in stake_values]

	for num_noiser in num_noisers:

		print(probabilities[line_idx])
		line = mlines.Line2D(stake_values, probabilities[line_idx], color=line_colors[line_idx], linewidth=3, linestyle='-', label=line_labels[line_idx])
		lines.append(line)
		line_idx = line_idx + 1

	for line in lines:
		ax.add_line(line)   

	plt.legend(handles=lines, loc='left', fontsize=18, title="Probability of successful collusion")

	axes = plt.gca()    

	plt.xlabel("Fraction of Adversarial Stake", fontsize=22)

	axes.set_xlim([0, 0.5])

	plt.ylabel("Probability of unmasked updates", fontsize=22)

	axes.set_ylim([0, 0.11])

	plt.yticks(np.arange(0, 0.11, 0.02))
	plt.xticks(np.arange(0, 0.51, 0.1))

	fig.tight_layout(pad=0.1)
	
	fig.savefig("eval_noise_attack.pdf")

if __name__ == '__main__':
	main()
