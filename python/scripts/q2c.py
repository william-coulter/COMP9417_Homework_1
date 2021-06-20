import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

penalties = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]

# Import
df = pd.read_csv("./data/transformed_data.csv")
X_columns = df.drop(labels="Y", axis="columns").columns

# Create a model for each penalty and store coeffs
coeffs = []
for penalty in penalties:
    model = Ridge(alpha=penalty)
    model.fit(df.drop(labels="Y", axis="columns"), df["Y"])
    coeffs.append(model.coef_)

# Convert list into np array so that we can transpose
np_coeffs = np.array(coeffs).T

# Plot the coef for each feature
colours = ["red", "brown", "green", "blue", "orange", "pink", "purple", "grey"]

print(coeffs)
ax = plt.gca()
ax.set_prop_cycle(color=colours)
ax.set_xscale("log")

for i, weights in enumerate(np_coeffs):
    ax.plot(penalties, weights, label=X_columns[i])

ax.legend()

plt.xlabel("penalties")
plt.ylabel("coefficients")
plt.title("Ridge coefficients for each penalty")
plt.savefig("./outputs/q2c.png")
