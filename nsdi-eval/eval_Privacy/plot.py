import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))
	
results = pd.read_csv('results.csv', header=None)

results.columns = ['ProbCollusion', 'ProbUnmask', 'NumNoisers']
averageResults = results.groupby(['ProbCollusion', 'NumNoisers'])['ProbUnmask'].mean().reset_index()

numNoisers = averageResults.NumNoisers.unique()

print numNoisers
print averageResults

for numNoise in numNoisers:

	plotValues = averageResults.loc[averageResults['NumNoisers'] == numNoise]
	thisLabel = "# Noisers = " + str(numNoise) 
	plt.plot(plotValues['ProbCollusion'],plotValues['ProbUnmask'], label= thisLabel, linewidth=5)

plt.legend(loc='best', fontsize=20)
plt.xlabel('Percentage of colluders in the system', fontsize=20)
plt.ylabel('Probability of unmasked updates', fontsize=20)

axes = plt.gca()
axes.set_xlim([0, 0.5])
axes.set_ylim([0, 0.1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("eval_noise_attack.pdf")

plt.show()