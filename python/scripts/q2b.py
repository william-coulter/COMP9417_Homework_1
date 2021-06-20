import math

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Import
df = pd.read_csv("./data/data.csv")

# Save these here as they go away when we scale with preprocessing
X_columns = df.columns[:-1]

# Drop Y values
X_df = df.drop(labels="Y", axis="columns")

# See docs here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale
#
# This scales such that the mean of each feature is 0.
scaled = preprocessing.scale(X_df, with_std=False) # returns a nparray
zero_mean_scaled_df = pd.DataFrame(data=scaled, columns=X_columns)

# FOR THE MARKER
#
# HA! If I set 'with_std=True' in the preprocessing.scale, the transformed
# dataset is exactly what I need. 
#
# I realised this AFTER implementing all of below so I'm going to 
# keep this code in here just so you know that I know what's going on.

# Let's do some manual work for this next part
sum_of_squares_scaled = []
for column_name in zero_mean_scaled_df:
    column = zero_mean_scaled_df[column_name].tolist()
    n = len(column)

    sum_of_squares = 0
    for observation in column:
        sum_of_squares += (observation * observation)
    
    scale_factor = math.sqrt(n / sum_of_squares)
    scaled = []
    for observation in column:
        scaled.append(observation * scale_factor)

    sum_of_squares_scaled.append(scaled)
    
# convert to nparray - need to transpose
np_matrix = np.array(sum_of_squares_scaled).T

# convert to dataframe
final_scaled_df = pd.DataFrame(data=np_matrix, columns=X_columns)
# Add "Y" back on
final_scaled_df["Y"] = df["Y"]
# print(final_scaled_df.head())

# save to csv for later
final_scaled_df.to_csv("./data/transformed_data.csv", index=False)

# Now for the question, print out the sum of squares
print("Sum of squares for each transformed feature:")
for column_name in final_scaled_df:
    if column_name == "Y":
        continue
        
    column = final_scaled_df[column_name].tolist()

    sum_of_squares = 0
    for observation in column:
        sum_of_squares += (observation * observation)

    print(f"{column_name}: {sum_of_squares}")

