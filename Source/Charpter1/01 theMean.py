import matplotlib.pyplot as plt
import math

def mean(sampleset):
    total = 0
    for element in sampleset:
        total = total + element

    return total/len(sampleset)


def variance(sampleset):
    total = 0
    setmean = mean(sampleset)
    for element in sampleset:
        total = total + math.pow(element-setmean, 2)
    return total/len(sampleset)


def standardDeviation(sampleset):
    total = 0
    setvariance = variance(sampleset)
    return math.sqrt(setvariance)


myset1 = [2., 10., 3., 6., 4., 6., 10.]
myset2 = [1., -100., 15., -100., 21.]
#mymean = mean(myset1)

# plt.isinteractive(block=False)
# plt.plot(myset)
# plt.plot([mymean] * 7)
# plt.show()

print("Variance of first set: {0} with standard deviation {1}".format(variance(myset1), standardDeviation(myset1)))
print("Variance of second set: {0} with standard deviation {1}".format(variance(myset2), standardDeviation(myset2)))
