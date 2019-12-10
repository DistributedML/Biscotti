import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


df = pd.read_csv("incremental_results.csv")
df = df.sort_values(by=['num_nodes'])
data = df.values

N = 4
width = 0.15
fig, ax = plt.subplots(figsize=(20, 10))

ticklabels = data[:,0]

data = data
print(data)


p1 = ax.bar(np.arange(N), data[:, 1], width, yerr=data[:,2], hatch='/')
p2 = ax.bar(np.arange(N) + 1 * width, data[:, 3], width, yerr=data[:,4])
p3 = ax.bar(np.arange(N) + 2 * width, data[:, 5], width, hatch='.', yerr=data[:,6])
p4 = ax.bar(np.arange(N) + 3 * width, data[:, 7], width, hatch='o', yerr=data[:,8])
p5 = ax.bar(np.arange(N) + 4 * width, data[:, 9], width, hatch='+', yerr=data[:,10])

# err = 1
# print(data[:2])

# p1 = ax.bar(np.arange(N), data[:, 1], width, yerr=data[:,2], hatch='/')
# p2 = ax.bar(np.arange(N) + 1 * width, data[:, 3], width, yerr=err)
# p3 = ax.bar(np.arange(N) + 2 * width, data[:, 5], width, hatch='.', yerr=err)
# p4 = ax.bar(np.arange(N) + 3 * width, data[:, 7], width, hatch='o', yerr=err)
# p5 = ax.bar(np.arange(N) + 4 * width, data[:, 9], width, hatch='+', yerr=err)

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

plt.xlabel('Number of Peers', fontsize=38)
plt.ylabel('Average Time Per Iteration (s)', fontsize=38)

ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0]),
          ('Noising', 'Verification', 'Secure Aggregation', "Flooding", 'Total'),
          loc='best', ncol=3, fontsize=28)

fig.tight_layout(pad=0.1)
fig.savefig("eval_cost_breakdown.pdf")

