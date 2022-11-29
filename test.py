import pandas as pd
import numpy as np
import plotly.express as px


df = pd.read_csv("houses_to_rent.csv", index_col=0)
# Computing the sample average
area_mean = np.mean(df["area"])

print("The original sample mean of the internal area of houses in Brazil is", area_mean)
print()
print("Rest of the dataset:")
df["area"]



