import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

	
results = pd.read_csv('results.csv', header=None)

results.columns = ['ProbCollusion', 'ProbUnmask', 'NumNoisers']
averageResults = results.groupby(['ProbCollusion', 'NumNoisers'])['ProbUnmask'].mean().reset_index()

numNoisers = averageResults.NumNoisers.unique()

print numNoisers
print averageResults

for numNoise in numNoisers:

	plotValues = averageResults.loc[averageResults['NumNoisers'] == numNoise]
	thisLabel = "NumNoisers = " + str(numNoise) 
	plt.plot(plotValues['ProbCollusion'],plotValues['ProbUnmask'], label= thisLabel)

plt.legend(loc='best')
plt.xlabel('Percentage of colluders in the system')
plt.ylabel('Probability of unmasked updates')
plt.show()