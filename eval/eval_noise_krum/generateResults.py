import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd
import matplotlib.lines as mlines

prefix = "epsilon"

eValues = ["001", "01", "05", "1", "2", "5"]
eLabels = ("0.01", "0.1","0.5", "1", "2", "5")

def getValues():

    outfile = open("noise_attackRate" +".csv", "w")

    towrite = ""

    count = 0

    for eValue in eValues:
        
        towrite = ""    

        fname = prefix+"_"+str(eValue)+".log"

        lines = [line.rstrip('\n') for line in open(fname)]

        idx  = -1

        attackRate = ""

        maxIteration = 96

        maxAttackRate = 0

        iterCount = 0

        for line in lines:
            
            idx = line.find("Attack Rate")    

            if idx != -1:
                attackRate = line[(idx + 15):(idx + 22)]
                iterCount = iterCount + 1

                if iterCount > maxIteration:
                    attackRate = float(line[(idx + 15):(idx + 22)])
                    if attackRate > maxAttackRate:
                        maxAttackRate = attackRate 

        towrite = eLabels[count] + "," + str(maxAttackRate) +"\n"    

        outfile.write(towrite)
        count = count + 1

    outfile.close()

def plot():

    fig, ax = plt.subplots(figsize=(10, 3))

    df = pd.read_csv("noise_attackRate.csv", header=None)    
    toplot = df.values

    index = np.arange(len(eLabels))

    print(toplot)

    ax = plt.axes()

    #For bar    
    thisLine =  ax.bar(index, toplot[:,1])     

    plt.xlabel(r'$\epsilon(Privacy Loss)$', fontsize=22)
    plt.ylabel("Attack Rate", fontsize=22)

    plt.xticks(index, eLabels) 

    plt.yticks(np.arange(0, 1.4, step=0.5)) 
   

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.tight_layout()
    fig.savefig("fig_noise_krum.pdf")

if __name__ == "__main__":

    getValues()
    plot()
