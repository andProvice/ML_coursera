import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

'''
Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv. 
Целевая переменная записана в первом столбце, признаки — во втором и третьем.
'''
train = pandas.read_csv('perceptron-train.csv', header=None)
test = pandas.read_csv('perceptron-test.csv', header=None)

X_train = train[train.columns[1:3]]
y_train = train[train.columns[0]]

X_test = test[test.columns[1:3]]
y_test = test[test.columns[0]]

'''
Обучите персептрон со стандартными параметрами и random_state=241
'''
perc = Perceptron()
clf = perc.fit(X_train, y_train)
X_train_predict = clf.predict(X_train)

X_test_predict = clf.predict(X_test)

'''
Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
'''
#X_train_accurance = accuracy_score(y_train, X_train_predict)
X_test_accurance = accuracy_score(y_test, X_test_predict)

#print('Train - ', X_train_accurance)
print('Test - ', X_test_accurance)

'''
Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler. 
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
'''
clf_norm = perc.fit(X_train_scaled, y_train)
#X_train_predict2 = clf_norm.predict(X_train_scaled)
#X_train_accurance2 = accuracy_score(y_train, X_train_predict2)

X_test_predict2 = clf_norm.predict(X_test_scaled)
X_test_accurance2 = accuracy_score(y_test, X_test_predict2)

#print('Train 2 - ', X_train_accurance2)
print('Test 2 - ', X_test_accurance2)

'''
Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. 
Это число и будет ответом на задание.
'''
'''
file = open('s6.txt', 'w')
file.write(str(round(X_test_accurance2 - X_test_accurance, 3)))
file.close()
'''
print(str(round(X_test_accurance2 - X_test_accurance, 3)))

print('fin')