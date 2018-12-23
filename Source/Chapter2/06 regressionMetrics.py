import math as math
import numpy as np
# Main Metrics

#   Mean absolute error (MAE)
def mean_absolute_error(y, yi):
    sum = 0
    for v,vi in y,yi:
        sum += math.fabs(v - vi)
    result = sum / len(y)
    return result


#   Median absolute error
def median_absolute_error(y, yi):
    return np.median(math.fabs(y-yi))


#   Mean squared error (MSE)
def mean_squred_error(y, yi):
    sum = 0
    for v, vi in y, yi:
        sum += math.sqrt(v - vi)
    result = sum / len(y)
    return result


# Classification Metrics
#   Accuracy
def accuracy(y, yi):
    sum = 0
    for v, vi in y, yi:
        sum += 1 if v == vi else 0
    result = sum / len(y)
    return result


#   Precision score, recall and F-measure
def precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)


def recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)


# prec =  precision, rec = recall
def f_measure(b2, prec, rec):
    return (1 + math.sqrt(b2))*(prec * rec) / (math.sqrt(b2) * prec + rec)

