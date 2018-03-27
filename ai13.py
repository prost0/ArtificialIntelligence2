import numpy as np
import pandas as pd
import patsy as pt
from sklearn import preprocessing
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from datetime  import datetime
import quandl

 
df = pd.read_csv(r"data_third.csv")
del df['Split Ratio']



df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'] - df['Date'][0]
df['Date'] = df['Date'].dt.days
dfn = (df - df.mean()) / (df.max() - df.min())#normalization
pt_y, pt_x = pt.dmatrices("Close ~ Open", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Close ~ Open ", b0)

"""
ax = plt.subplot()
scatter_matrix(df, alpha=0.05, figsize=(10, 10), marker ='x')

ax.plot(dfn['Close'], dfn['Open'], 'go', color = 'blue')#x[x]
axis_x = np.linspace(-1, 1, 100)
f = b0[0] + b0[1] * axis_x 
ax.plot(axis_x, f, color = 'red')
"""
ax = plt.subplot()
plt.scatter(dfn['Date'], dfn['Close'], c='blue', s=20, label='blue',
                alpha=0.3, edgecolors='none')

pt_y, pt_x = pt.dmatrices("Close ~ Date", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
axis_x = np.linspace(-1, 1, 100)
f = b0[0] + b0[1] * axis_x
ax.plot(axis_x, f, color = 'red')

x = dfn.iloc[:,:-1]
y = dfn.iloc[:,-1]

pt_y, pt_x = pt.dmatrices("y ~ x", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b1 = res[0].ravel()
print ("Adj. Volume ~ <Others> ", b1)

plt.show()
