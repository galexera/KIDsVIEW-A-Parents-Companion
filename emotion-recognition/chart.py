import pandas as pd
from matplotlib.pyplot import pie, axis, show
import matplotlib.animation as animation
import time

df = pd.read_csv ('Report/analysis.csv')

sums = df.groupby(df["Emotion"])["Value"].sum()
axis('equal')
pie(sums, labels=sums.index)


show()
