import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

prefix = "poison"
poisoningValues_1 = ["50%", "45%", "40%", "30%", "10%"]
poisoningValues = ["0.50", "0.45", "0.40", "0.30", "0.10"]
samples = ["20", "40", "70"]
samples_1 = ["20%", "40%", "70%"]

def plot():

    fig, ax = plt.subplots(figsize=(10, 5))

    df = pd.read_csv("heatmap.csv", header=None)
    
    toplot = df.values
    toplot = np.around(toplot, decimals=2)
    print(toplot)


    ax = sns.heatmap(toplot, mask=np.zeros((5,3)), 
        linewidth=0.5, annot=True, annot_kws={"size": 20, 'weight':'bold'}, fmt=".2f",
        center=0, cmap='Greys', xticklabels=samples_1, yticklabels=poisoningValues_1, cbar_kws={'label': '1-7 Attack Rate'})

    ax.figure.axes[-1].yaxis.label.set_size(22)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=18)


    plt.xlabel("Number (%) of Received Updates", fontsize=22)
    plt.ylabel("Percent of Poisoners", fontsize=22)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.tight_layout()

    plt.tight_layout(pad=0.1)
    fig.savefig("fig_poison_nsamples.pdf")


def getHeatMapValues():

    outfile = open("heatmap" +".csv", "w")

    for poisonValue in poisoningValues:
        
        towrite = ""

        for sample in samples:

            fname = prefix+"_"+str(poisonValue)+"_"+str(sample)+".log"
            lines = [line.rstrip('\n') for line in open(fname)]
            idx  = -1
            
            attackRate = ""

            maxIteration = 80

            maxAttackRate = 0

            iterCount = 0

            for line in lines:
                
                idx = line.find("Attack Rate")    

                if idx != -1:
                    attackRate = line[(idx + 15):(idx + 22)]
                    iterCount = iterCount + 1

                if iterCount > maxIteration:
                    attackRate = float(attackRate)
                    if attackRate > maxAttackRate:
                        maxAttackRate = attackRate 
            
            if sample == "70":
                towrite = towrite+str(maxAttackRate)+"\n"
            else:
                towrite = towrite+str(maxAttackRate)+","

        outfile.write(towrite)

    outfile.close()

if __name__ == "__main__":

    # getHeatMapValues()
    plot()