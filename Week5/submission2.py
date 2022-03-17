import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

'''
Градиентный бустинг над решающими деревьями
'''

'''
Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy (параметр values у датафрейма). 
В первой колонке файла с данными записано, была или нет реакция. Все остальные колонки (d1 - d1776) 
содержат различные характеристики молекулы, такие как размер, форма и т.д. 
Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241. 
'''
data = pandas.read_csv('gbm-data.csv')
data = data.values

y = data[:, [0]]
X = data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 241)

'''
Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 
и для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:

    Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке на каждой итерации.
    
    Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}), 
    где y_pred — предсказанное значение.
    
    Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции sklearn.metrics.log_loss) 
    на обучающей и тестовой выборках, а также найдите минимальное значение метрики и номер итерации, на которой оно достигается.
'''
kRFC = 0
plt.figure()
for i in [1, 0.5, 0.3, 0.2, 0.1]:
    GBC = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
    GBC.fit(X_train, y_train)
    test_loss = np.empty(250)
    for m, y_pred in enumerate(GBC.staged_decision_function(X_test)):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        test_loss[m] = log_loss(y_test, y_pred)
    print(str(test_loss.argmin()) + ' ' + str(test_loss[test_loss.argmin()].round(3)))
    '''
    Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается, 
    при learning_rate = 0.2.
    '''
    if i == 0.2:
        kRFC = test_loss.argmin()
        file = open('s3.txt', 'w')
        file.write(str(test_loss[test_loss.argmin()].round(2)) + ' ' + str(test_loss.argmin()))
        file.close()


    train_loss = np.empty(250)
    for m, y_pred in enumerate(GBC.staged_decision_function(X_train)):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        train_loss[m] = log_loss(y_train, y_pred)

    plt.plot(test_loss, color=(i, 0, abs(1-i)), label=str(i) + ' Test')
    plt.plot(train_loss, '--', color=(i, 0, i), label=str(i) + ' Train')

plt.legend()
plt.show()

'''
Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации: 
переобучение (overfitting) или недообучение (underfitting)? 
В ответе укажите одно из слов overfitting либо underfitting.
'''
file = open('s2.txt', 'w')
file.write('overfitting')
file.close()

'''
На этих же данных обучите RandomForestClassifier с количеством деревьев, 
равным количеству итераций, на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, 
c random_state=241 и остальными параметрами по умолчанию. 
Какое значение log-loss на тесте получается у этого случайного леса? 
(Не забывайте, что предсказания нужно получать с помощью функции predict_proba. 
В данном случае брать сигмоиду от оценки вероятности класса не нужно)
'''
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=kRFC, random_state=241)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)
res = log_loss(y_test, y_pred)
#print(res)

file = open('s4.txt', 'w')
file.write(str(round(res, 2)))
file.close()

print('fin')