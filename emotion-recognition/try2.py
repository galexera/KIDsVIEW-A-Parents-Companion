import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.pyplot import pie, axis, show

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
fig, ax = plt.subplots()

def update(i):
    # pullData = open("sample.txt","r").read()
    df = pd.read_csv ('Report/analysis.csv')
    dataArray = df.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
                
            sums = df.groupby(df["emotion"])["value"].sum()
            axis('equal')
    pie(sums, labels=sums.index)

ani = FuncAnimation(fig, update, frames=range(100), repeat=False)
plt.show()