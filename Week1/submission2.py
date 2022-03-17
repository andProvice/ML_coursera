import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

'''
В этом задании мы вновь рассмотрим данные о пассажирах Титаника. 
Будем решать на них задачу классификации, в которой по различным характеристикам пассажиров требуется предсказать, кто из них выжил после крушения корабля. 

Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
Обратите внимание, что признак Sex имеет строковые значения.
Выделите целевую переменную — она записана в столбце Survived.
В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. 
Такие записи при чтении их в pandas принимают значение nan. 
Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
Вычислите важности признаков и найдите два признака с наибольшей важностью. 
Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен). 
'''

data = pandas.read_csv('titanic/train.csv', index_col='PassengerId')
data['Sex'] = data['Sex'].map({'female':0, 'male':1})
x = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
y = x.pop('Survived')

model = DecisionTreeClassifier(random_state=241)
model.fit(x, y)

#print(x.head(5))
print(" ".join(pandas.Series(model.feature_importances_, index=x.columns).nlargest(2).index.values.tolist()))

'''
file7 = open('s7.txt', 'w')
file7.write((" ".join(pandas.Series(model.feature_importances_, index=x.columns).nlargest(2).index.values.tolist())))
file7.close()
'''

print('fin')
