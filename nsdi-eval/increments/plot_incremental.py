import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


df = pd.read_csv("incremental_results.csv", header=None)
data = df.values

# plt.plot(data1[0], np.mean(data1[1:3], axis=0), color="black", label="MNIST", lw=3)
# plt.plot(data2[0], np.mean(data2[1:3], axis=0), color="red", label="KDDCup", lw=3)
# plt.plot(data3[0], np.mean(data3[1:3], axis=0), color="orange", label="Amazon", lw=3)

N = 4
width = 0.20
fig, ax = plt.subplots(figsize=(10, 5))

ticklabels = ['40', '60', '80', '100']

data = data / 100.0

# Take the reverse
for i in range(1,4):
	data[:, i] = (data[:, -1] - data[:, i])

p1 = ax.bar(np.arange(N), data[:, 1], width, hatch='/')
p2 = ax.bar(np.arange(N) + width, data[:, 2], width)
p3 = ax.bar(np.arange(N) + 2 * width, data[:, 3], width, hatch='.')
p4 = ax.bar(np.arange(N) + 3 * width, data[:, 4], width, hatch='+')

ax.set_xticks(np.arange(N) + 1.5 * width)
ax.set_xticklabels(ticklabels, fontsize=28)

# ax.set_yticklabels(np.arange(0, 7, 1))
plt.setp(ax.get_yticklabels(), fontsize=28)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0, 100)

totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    height = i.get_height()
    if height < 1:
    	ax.text(i.get_x(), i.get_height() + 2, str(height)[0:3] + " ", fontsize=26, color='black')
    elif height < 10:
    	ax.text(i.get_x(), i.get_height() + 2, str(height)[0:3], fontsize=26, color='black')
    else:
    	ax.text(i.get_x(), i.get_height() + 2, str(height)[0:4], fontsize=26, color='black')

# ##############################

plt.xlabel('Number of Peers', fontsize=32)
plt.ylabel('Time Per Iteration (s)', fontsize=32)

ax.legend((p1[0], p2[0], p3[0], p4[0]),
          ('Noising', 'Verification', 'Secure Aggregation', 'Total'),
          loc='best', ncol=2, fontsize=28)

fig.tight_layout(pad=0.1)
fig.savefig("eval_cost_breakdown.pdf")

