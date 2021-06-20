import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./data/data.csv")
sns.pairplot(df)
plt.savefig("./outputs/q2a_pairplots.png")