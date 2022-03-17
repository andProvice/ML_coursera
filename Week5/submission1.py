import pandas
import numpy as np
from sklearn.decomposition import PCA

'''
Размер случайного леса
'''

'''
Загрузите данные из файла abalone.csv. 
Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
'''
data = pandas.read_csv('abalone.csv')

'''
Преобразуйте признак Sex в числовой: 
значение F должно перейти в -1, I — в 0, M — в 1. 
Если вы используете Pandas, то подойдет следующий код: 
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
'''
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

'''
Разделите содержимое файлов на признаки и целевую переменную. 
В последнем столбце записана целевая переменная, в остальных — признаки.
'''
y = data[data.columns[-1:]]
x = data[data.columns[:-1]]

'''
Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: 
от 1 до 50 (не забудьте выставить "random_state=1" в конструкторе). 
Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам. 
Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold. 
В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

gen = KFold(n_splits=5, shuffle=True, random_state=1)

res = {}
for i in range(1, 51):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    quality = cross_val_score(X=x, y=y, estimator=clf, cv=gen, scoring='r2')
    res[i] = quality.mean().round(2)

'''
Определите, при каком минимальном количестве деревьев случайный лес показывает качество на кросс-валидации выше 0.52. 
Это количество и будет ответом на задание.
'''
for i in range(1, 51):
    if res[i] > 0.52:
        break
print(i)
print(res[i])

file = open('s1.txt', 'w')
file.write(str(i))
file.close()

print('fin')