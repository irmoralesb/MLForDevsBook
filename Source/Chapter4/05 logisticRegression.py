import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import linear_model
import seaborn.apionly as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid', context='notebook')

# Displaying original data
df = pd.read_csv("data/CHD.csv", header=0)
plt.figure()
plt.axis([0, 70, -0.2, 1.2])
plt.title("Original data")
plt.scatter(df['age'], df['chd'])  # Plot a scatter draw of the random data points
plt.show()

# Creating logistic regression model
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(df['age'].values.reshape(100, 1), df['chd'].values.reshape(100, 1))

linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                fit_intercept=True, intercept_scaling=1, max_iter=100,
                                multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                                solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

x_plot = np.linspace(10, 90, 100)
oneprob = []
zeroprob = []
predict = []
plt.figure(figsize=(10, 10))
for i in x_plot:
    oneprob.append(logistic.predict_proba(i)[0][1])
    zeroprob.append(logistic.predict_proba(i)[0][0])
    predict.append(logistic.predict(i)[0])

plt.plot(x_plot, oneprob)
plt.plot(x_plot, zeroprob)
plt.plot(x_plot, predict)
plt.scatter(df['age'], df['chd'])
plt.show()

# Getting Error
# /usr/bin/python3.6 /opt/pycharm-2018.3.2/helpers/pydev/pydevconsole.py --mode=client --port=39107
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/irmorales/Projects/MLForDevsBook/Source'])
# PyDev console: starting.
# Python 3.6.5 (default, Mar 31 2018, 19:45:04) [GCC] on linux
# runfile('/home/irmorales/Projects/MLForDevsBook/Source/Chapter4/05 logisticRegression.py', wdir='/home/irmorales/Projects/MLForDevsBook/Source/Chapter4')
# /usr/lib64/python3.6/site-packages/matplotlib/__init__.py:855: MatplotlibDeprecationWarning:
# examples.directory is deprecated; in the future, examples will be found relative to the 'datapath' directory.
#   "found relative to the 'datapath' directory.".format(key))
# /usr/lib64/python3.6/site-packages/matplotlib/__init__.py:846: MatplotlibDeprecationWarning:
# The text.latex.unicode rcparam was deprecated in Matplotlib 2.2 and will be removed in 3.1.
#   "2.2", name=key, obj_type="rcparam", addendum=addendum)
# /usr/lib/python3.6/site-packages/seaborn/apionly.py:9: UserWarning: As seaborn no longer sets a default style on import, the seaborn.apionly module is deprecated. It will be removed in a future version.
#   warnings.warn(msg, UserWarning)
# /usr/lib64/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
#   FutureWarning)
# /usr/lib64/python3.6/site-packages/sklearn/utils/validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#   y = column_or_1d(y, warn=True)
# Traceback (most recent call last):
#   File "<input>", line 1, in <module>
#   File "/opt/pycharm-2018.3.2/helpers/pydev/_pydev_bundle/pydev_umd.py", line 198, in runfile
#     pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
#   File "/opt/pycharm-2018.3.2/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
#     exec(compile(contents+"\n", file, 'exec'), glob, loc)
#   File "/home/irmorales/Projects/MLForDevsBook/Source/Chapter4/05 logisticRegression.py", line 33, in <module>
#     oneprob.append(logistic.predict_proba(i)[0][1]);
#   File "/usr/lib64/python3.6/site-packages/sklearn/linear_model/logistic.py", line 1411, in predict_proba
#     return super(LogisticRegression, self)._predict_proba_lr(X)
#   File "/usr/lib64/python3.6/site-packages/sklearn/linear_model/base.py", line 295, in _predict_proba_lr
#     prob = self.decision_function(X)
#   File "/usr/lib64/python3.6/site-packages/sklearn/linear_model/base.py", line 257, in decision_function
#     X = check_array(X, accept_sparse='csr')
#   File "/usr/lib64/python3.6/site-packages/sklearn/utils/validation.py", line 545, in check_array
#     "if it contains a single sample.".format(array))
# ValueError: Expected 2D array, got scalar array instead:
# array=10.0.
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
