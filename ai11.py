import numpy as np
import pandas as pd
import patsy as pt
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"challenge_dataset.txt")
df = (df - df.mean()) / df.std()
train, test = train_test_split(df, test_size=0.2)

x = train.iloc[:,:-1]
y = train.iloc[:,-1]
x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

pt_y, pt_x = pt.dmatrices("y ~ x", train)
res = np.linalg.lstsq(pt_x, pt_y)
b = res[0].ravel()

plt.subplot(211)
plt.plot(x, y, 'go', color = 'blue')
x2 = np.linspace(-4, 4, 100)
f1 = b[0] + b[1] * x2 
plt.plot(x2, f1, color = 'red')

plt.subplot(212)
plt.plot(x_test, y_test, 'go', color = 'blue')
f2 = b[0] + b[1] * x_test
plt.plot(x_test, f2, color = 'red')

pt_y, pt_x = pt.dmatrices("y_test ~ x_test", test)
res = np.linalg.lstsq(pt_x, pt_y)
b = res[0].ravel()

x2 = np.linspace(-1, 4, 100)
f1 = b[0] + b[1] * x2 
plt.plot(x2, f1, color = 'green')
print (b)
plt.show()
