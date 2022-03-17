import pandas
import numpy as np
from sklearn.decomposition import PCA

'''
Составление фондового индекса
'''

'''
Загрузите данные close_prices.csv. 
В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода. 
'''
train = pandas.read_csv('close_prices.csv')
y = train[train.columns[0]]
x = train[train.columns[1:]]

'''
На загруженных данных обучите преобразование PCA с числом компоненты равным 10. 
Скольких компонент хватит, чтобы объяснить 90% дисперсии?
'''
cls = PCA(n_components=10)
cls.fit(x)

sum = 0
for i in range(len(cls.explained_variance_ratio_)):
    sum += cls.explained_variance_ratio_[i]
    if sum > 0.9:
        break
print(i+1)
#print(cls.explained_variance_ratio_)

file = open('s2.txt', 'w')
file.write(str(i+1))
file.close()

'''
Примените построенное преобразование к исходным данным и возьмите значения первой компоненты. 
'''
res = pandas.DataFrame(cls.transform(x)[:, 0])
#print(res)

'''
Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. 
Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса? 
'''
test = pandas.read_csv('djia_index.csv')

corr = np.corrcoef(res.T, test['^DJI'].T)[1, 0]
print(corr.round(2))

file = open('s3.txt', 'w')
file.write(str(corr.round(2)))
file.close()

'''
Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
'''
print(train.columns[np.argmax(cls.components_[0]) + 1])

file = open('s4.txt', 'w')
file.write(train.columns[np.argmax(cls.components_[0]) + 1])
file.close()

print('fin')