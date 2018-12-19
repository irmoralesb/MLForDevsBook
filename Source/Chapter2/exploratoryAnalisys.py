import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/iris.csv")
print(df.columns)
print(df.head(3))
print(df["Sepal.Length"].head(3))

#Descrive the sepal length column
print("Mean: {0}".format(df["Sepal.Length"].mean()))
print("Standard deviation: {}".format(df["Sepal.Length"].std()))
print("Kurtosis: {}".format(df["Sepal.Length"].kurtosis()))
print("Skewness: {}".format(df["Sepal.Length"].skew()))

#df["Sepal.Length"].plot.hist()
plt.hist(df["Sepal.Length"])
plt.show()

# Observations:
# Skew is positive, which indicates the lateral deviation is to the left.
# Kurtosis is negative, which indicates the degree is low (peakedness is low), so the values are distributed,
# not concentrated
