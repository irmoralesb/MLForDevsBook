from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/mpg.csv")
plt.figure(figsize=(10, 8))
print(df.columns)
partialcolumns = df[['acceleration', 'mpg']]
std_scale = preprocessing.StandardScaler().fit(partialcolumns)
df_std = std_scale.transform(partialcolumns)
plt.scatter(partialcolumns['acceleration'], partialcolumns['mpg'], color="grey", marker="^")
plt.scatter(df_std[:, 0], df_std[:, 1])
plt.show()
