import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


df = pd.read_csv("eval_vrf.csv")
data = df.values

# plt.plot(data1[0], np.mean(data1[1:3], axis=0), color="black", label="MNIST", lw=3)
# plt.plot(data2[0], np.mean(data2[1:3], axis=0), color="red", label="KDDCup", lw=3)
# plt.plot(data3[0], np.mean(data3[1:3], axis=0), color="orange", label="Amazon", lw=3)

N = 3
width = 0.20
fig, ax = plt.subplots(figsize=(10, 5))

ticklabels = ['Noisers', 'Verifiers', 'Aggregators']

p1 = ax.bar(np.arange(N), data[:,1], yerr=data[:,2] , width=width, color='purple')
p2 = ax.bar(np.arange(N) + width, data[:,3], yerr=data[:,4], width=width, color='lightgreen', hatch='/')
p3 = ax.bar(np.arange(N) + 2 * width, data[:,5], yerr=data[:,6], width=width, color='orange', hatch='.')
p4 = ax.bar(np.arange(N) + 3 * width, data[:,7], yerr=data[:,8], width=width, color='blue', hatch='.')

ax.set_xticks(np.arange(N) + width)
ax.set_xticklabels(ticklabels, fontsize=22)

# ax.set_yticklabels(np.arange(0, 7, 1))
plt.setp(ax.get_yticklabels(), fontsize=22)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0, 90)

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
    ax.text(i.get_x(), i.get_height() + 2, " " + str(height)[0:4], fontsize=16, color='black')

# ##############################

plt.yticks((20, 40, 60, 80, 100, 120))

plt.ylabel('Avg Time Per Iteration (s)', fontsize=28)

ax.legend((p1[0], p2[0], p3[0], p4[0]),
          ('N = 3', 'N = 5', 'N = 10', 'N=26'),
          loc='upper center', ncol=3, fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("eval_nva_scale.pdf")
plt.show()
