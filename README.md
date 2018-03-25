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
```
```python
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
