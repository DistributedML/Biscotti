import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

prefix = "poison"
poisoningValues = ["50%", "45%", "40%", "30%", "10%"]
samples = ["20(21%)", "40(41%)", "70(75%)"]

def plot():

    fig, ax = plt.subplots(figsize=(10, 5))

    df = pd.read_csv("heatmap.csv", header=None)
    
    toplot = df.values
    toplot = np.around(toplot, decimals=2)
    print(toplot)


    ax = sns.heatmap(toplot, mask=np.zeros((5,3)), 
        linewidth=0.5, annot=True, annot_kws={"size": 20, 'weight':'bold'}, fmt=".2f",
        center=0, cmap='Greys', xticklabels=samples, yticklabels=poisoningValues, cbar_kws={'label': '1-7 Attack Rate'})

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
            for line in lines:
                
                idx = line.find("Attack Rate")    

                if idx != -1:
                    attackRate = line[(idx + 15):(idx + 22)]
            
            if sample == "70":
                towrite = towrite+attackRate+"\n"
            else:
                towrite = towrite+attackRate+","

        outfile.write(towrite)

    outfile.close()

if __name__ == "__main__":

    # getHeatMapValues()
    plot()