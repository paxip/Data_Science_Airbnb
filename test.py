import pandas as pd
import numpy as np
import plotly.express as px



# df = pd.read_csv("EntriesGender.csv")
# fig = px.bar(df, x='Discipline', y="Total")
# fig.show()

# df = pd.read_csv("EntriesGender.csv")
# fig = px.bar(df, x=("Male", "Female"), y=("Total"))
# fig.show()



# df = pd.read_csv("EntriesGender.csv")
# print(df.head())

# fig = px.bar(df, x = 'Discipline', y = ['Female', 'Male'], color='Total')
# fig.show()

# df = pd.read_csv('Athletes.csv', encoding= 'unicode_escape')
# df = pd.read_csv('Athletes.csv', index_col=0)
# print(df.head())


# # Here we use a column with categorical data
# fig = px.histogram(df, x="NOC")
# fig.show()

df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/Medals.csv', encoding= 'unicode_escape')
print(df.head())
long_df = px.data.medals_long()
fig = px.bar(df, x="Team/NOC", y="Total", color=('Gold', 'Silver', 'Bronze'), title="Long-Form Input")
fig.show()

