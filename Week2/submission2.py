import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

'''
Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). 
Результатом вызова данной функции является объект, у которого признаки записаны в поле data, а целевой вектор — в поле target.
'''
data = load_boston()
y = data['target']
x = data['data']

'''
Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
'''
x = scale(x)

'''
Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, 
чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace). 
Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, 
зависящие от расстояния до ближайших соседей. 
В качестве метрики качества используйте среднеквадратичную ошибку 
(параметр scoring='mean_squared_error' у cross_val_score; 
при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error'). 
Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42, 
не забудьте включить перемешивание выборки (shuffle=True).
'''
gen = KFold(n_splits=5, shuffle=True, random_state=42)

res = {}
for k in np.linspace(1, 10, 200):
    kv =  KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=k)
    quality = cross_val_score(kv, x, y, cv=gen, scoring='neg_mean_squared_error')
    quality = np.mean(quality)
    res[k] = quality

'''
Определите, при каком p качество на кросс-валидации оказалось оптимальным. 
Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать среднее этих показателей. 
Это значение параметра и будет ответом на задачу.
'''
p_max = 1
res_max = res[p_max]

for k in res.keys():
    if res_max <= res[k]:
        p_max = k
        res_max = res[k]
'''
file = open('s5.txt', 'w')
file.write(str(round(p_max, 1)))
file.close()
'''
print(p_max, res_max)

print('fin')
