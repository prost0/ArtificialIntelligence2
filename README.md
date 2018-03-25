Загрузка необходимых пакетов
```python
import numpy as np
import pandas as pd
import patsy as pt
from sklearn import preprocessing
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from datetime  import datetime
import quandl
```
challenge_dataset.txt
==
Загрузим датасет challenge_dataset.txt:
```python
  df = pd.read_csv(r"challenge_dataset.txt")
```
стандартизуем данные:
```python
  df = pd.read_csv(r"challenge_dataset.txt")
```
построим график:
```python
  ax = plt.subplot()
  ax.plot(x, y, 'go', color = 'blue')
  plt.show()
```
Для этих данных построим простую линейную регрессию y от x
```python
  x = df.iloc[:,:-1]
  y = df.iloc[:,-1]

  pt_y, pt_x = pt.dmatrices("y ~ x", df)
  res = np.linalg.lstsq(pt_x, pt_y)
  b = res[0].ravel()
```
построим полученную линию 
```python
  x2 = np.linspace(-4, 4, 100)
  f1 = b[0] + b[1] * x2 
  ax.plot(x2, f1, color = 'red')
```
![](pngs/ai1.png)

global_co2.csv
==
Загрузим датасет и стандартизуем данные:
```python
df = pd.read_csv(r"global_co2.csv")
df = (df - df.mean()) / df.std()
```

Будем предсказать параметр ‘Per capita’. Т.к. в этом параметре есть NA, то необходимо сначала отчистить данные

WIKI/GOOGL
==
Загрузим датасет, удалим  и заменим даты на количество дней с начала торгов и нормализуем данные:

```python
df = quandl.get("WIKI/GOOGL")
del df['Split Ratio']

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'] - df['Date'][0]
df['Date'] = df['Date'].dt.days
dfn = (df - df.mean()) / (df.max() - df.min())
```

Построим scatter_matrix, чтобы увидеть зависимости:
```python
ax = plt.subplot()
scatter_matrix(df, alpha=0.05, figsize=(10, 10), marker ='x')
```
![](pngs/ai13scatterMatrix.png)

Построим линейную зависимость Close от Open:
```python
pt_y, pt_x = pt.dmatrices("Close ~ Open", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Close ~ Open ", b0))
```
Построим линейную зависимость Close от Date:
```python
pt_y, pt_x = pt.dmatrices("Close ~ Open", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
print ("Close ~ Open ", b0))
```

Построим график:
```python
ax.plot(dfn['Close'], dfn['Open'], 'go', color = 'blue')#x[x]
axis_x = np.linspace(-1, 1, 100)
f = b0[0] + b0[1] * axis_x 
ax.plot(axis_x, f, color = 'red')
```
![](pngs/ai3closeOpen.png)

Построим график линейной зависимости Close от Date:
```python
pt_y, pt_x = pt.dmatrices("Close ~ Date", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b0 = res[0].ravel()
axis_x = np.linspace(-1, 1, 100)
f = b0[0] + b0[1] * axis_x
ax.plot(axis_x, f, color = 'red')
```
![](pngs/ai13closeDate.png)

Найдем вектор коофициентов линейной зависимости Adj. Volume от остальных признаков:
```python
x = dfn.iloc[:,:-1]
y = dfn.iloc[:,-1]

pt_y, pt_x = pt.dmatrices("y ~ x", dfn)
res = np.linalg.lstsq(pt_x, pt_y)
b1 = res[0].ravel()
print ("Adj. Volume ~ <Others> ", b1)
```



