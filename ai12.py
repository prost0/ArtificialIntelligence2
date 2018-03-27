import numpy as np
import pandas as pd
import patsy as pt
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv(r"global_co2.csv")
df = (df - df.mean()) / df.std()

#scatter_matrix(df, alpha=0.05, figsize=(10, 10), marker ='x')
#plt.savefig("plot.png")
df1 = df.dropna(axis=1)                     #delete cols
df2 = df.dropna(axis=0)                     #delete rows
df3 = df.fillna(df['PerCapita'].mean())  #fill with median

x0 = df.iloc[:,:-1]
y0 = df.iloc[:,-1]
x1 = df1.iloc[:,:-1]
y1 = df1.iloc[:,-1]
x2 = df2.iloc[:,:-1]
y2 = df2.iloc[:,-1]
x3 = df3.iloc[:,:-1]
y3 = df3.iloc[:,-1]

pt_y, pt_x = pt.dmatrices("PerCapita ~ Year", df)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Deafault (PC ~ SF) ", b0)


pt_y, pt_x = pt.dmatrices("PerCapita ~ Year", df2)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Cross off rows with NAN (PC ~ Year) ", b0)



pt_y, pt_x = pt.dmatrices("PerCapita ~ Year", df3)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Fill NAN with median  (PC ~ Year) ", b0)


ax = plt.subplot()
ax.plot(df3['SolidFuel'], df3['PerCapita'], 'go', color = 'blue')#x[x]
asix_x = np.linspace(0, 4, 100)
f = b0[0] + b0[1] * asix_x 
ax.plot(asix_x, f, color = 'red')


pt_y, pt_x = pt.dmatrices("PerCapita ~ Year + Total + GasFuel + LiquidFuel + SolidFuel + Cement + GasFlaring", df)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Deafault ", b0)

pt_y, pt_x = pt.dmatrices("y1 ~ x1", df1)
res = np.linalg.lstsq(pt_x, pt_y)
b1 = res[0].ravel()
print ("Cross off cols with NAN ", b1)

pt_y, pt_x = pt.dmatrices("y2 ~ x2", df2)
res = np.linalg.lstsq(pt_x, pt_y)
b2 = res[0].ravel()
print ("Cross off rows with NAN ", b2)

pt_y, pt_x = pt.dmatrices("y3 ~ x3", df3)
res = np.linalg.lstsq(pt_x, pt_y)
b3 = res[0].ravel()
print ("Fill NAN with median ", b3)

plt.show()
