import pandas
import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

data = np.genfromtxt('wine.data', delimiter=',')
y = data[:,0]
x = data[:,1:]

'''
Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). 
Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). 
Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42. 
В качестве меры качества используйте долю верных ответов (accuracy).
'''
gen = KFold(n_splits=5, shuffle=True, random_state=42)

'''
Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. 
При каком k получилось оптимальное качество? 
Чему оно равно (число в интервале от 0 до 1)? 
Данные результаты и будут ответами на вопросы 1 и 2.
'''
res = {}
for k in range(1, 51):
    kv = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(kv, x, y, cv=gen, scoring='accuracy')
    quality = np.mean(quality)
    res[k] =quality

k_max = 0
res_max = 0

for k in range(1, 51):
    if res_max <= res[k]:
        k_max = k
        res_max = res[k]

file = open('s1.txt', 'w')
file.write(str(k_max))
file.close()

file = open('s2.txt', 'w')
file.write(str(round(res_max, 2)))
file.close()

print(k_max, res_max)

'''
Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. 
Снова найдите оптимальное k на кросс-валидации.
'''
x = scale(x)

'''
Какое значение k получилось оптимальным после приведения признаков к одному масштабу? 
Приведите ответы на вопросы 3 и 4. 
Помогло ли масштабирование признаков?
'''
res = {}
for k in range(1, 51):
    kv = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(kv, x, y, cv=gen, scoring='accuracy')
    quality = np.mean(quality)
    res[k] =quality

k_max = 0
res_max = 0

for k in range(1, 51):
    if res_max <= res[k]:
        k_max = k
        res_max = res[k]

file = open('s3.txt', 'w')
file.write(str(k_max))
file.close()

file = open('s4.txt', 'w')
file.write(str(round(res_max, 2)))
file.close()

print(k_max, res_max)

print('fin')
