import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets


'''
Для начала вам потребуется загрузить данные. 
В этом задании мы воспользуемся одним из датасетов, доступных в scikit-learn'е — 20 newsgroups. 
Для этого нужно воспользоваться модулем datasets:
После выполнения этого кода массив с текстами будет находиться в поле newsgroups.data, номер класса — в поле newsgroups.target.

Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм" (инструкция приведена выше). 
Обратите внимание, что загрузка данных может занять несколько минут
'''
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

'''
Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление. 
Одним из способов нахождения такого представления является вычисление TF-IDF. 
В Scikit-Learn это реализовано в классе sklearn.feature_extraction.text.TfidfVectorizer. 
Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.

Вычислите TF-IDF-признаки для всех текстов. 
Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным. 
При таком подходе получается, что признаки на обучающем множестве используют информацию из тестовой выборки — но такая ситуация вполне законна, 
поскольку мы не используем значения целевой переменной из теста. 
На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения, 
и поэтому можно ими пользоваться при обучении алгоритма.
'''
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

'''
Реализация SVM-классификатора находится в классе sklearn.svm.SVC. 
Веса каждого признака у обученного классификатора хранятся в поле coef_. 
Чтобы понять, какому слову соответствует i-й признак, можно воспользоваться методом get_feature_names() у TfidfVectorizer:

Подбор параметров удобно делать с помощью класса sklearn.grid_search.GridSearchCV 
(При использовании библиотеки scikit-learn версии 18.0.1 sklearn.model_selection.GridSearchCV). 

Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') 
при помощи кросс-валидации по 5 блокам. 
Укажите параметр random_state=241 и для SVM, и для KFold. 
В качестве меры качества используйте долю верных ответов (accuracy).
'''

kf = KFold(n_splits=5, shuffle=True, random_state=241)

grid = {'C':np.power(10.0, np.arange(-5,6))}
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
gs.fit(x, y)
'''
Первым аргументом в GridSearchCV передается классификатор, для которого будут подбираться значения параметров, 
вторым — словарь (dict), задающий сетку параметров для перебора. 
После того, как перебор окончен, можно проанализировать значения качества для всех значений параметров и выбрать наилучший вариант:
'''

#print(gs.best_params_['C'])

'''
Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
'''
clf = SVC(C=gs.best_params_['C'], kernel='linear', random_state=241)
clf.fit(x, y)

'''
Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC). 
Они являются ответом на это задание. 
Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке
'''
words = []
for i in np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]:
    words.append(vectorizer.get_feature_names()[i])

print(words)

file = open('s2.txt', 'w')
file.write(','.join(sorted(words)))
file.close()

print('fin')