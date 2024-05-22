import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("Optimal_LC.csv")

df["reward"].plot()

plt.ylim(0, 1)

plt.savefig("optimal.pdf")