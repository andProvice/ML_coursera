import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

'''
Линейная регрессия: прогноз оклада по описанию вакансии
'''

'''
Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv 
(либо его заархивированную версию salary-train.zip).
'''
train = pandas.read_csv('salary-train.csv')
test = pandas.read_csv('salary-test-mini.csv')

'''
Проведите предобработку:
    Приведите тексты к нижнему регистру (text.lower()).
'''
train['FullDescription'] = train.FullDescription.str.lower()

'''
    Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. 
    Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
    Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
'''
train['FullDescription'] = train.FullDescription.replace('[^a-zA-Z0-9]', ' ', regex = True)
#print(train['FullDescription'])
'''
Примените TfidfVectorizer для преобразования текстов в векторы признаков. 
Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
'''
vectorizer = TfidfVectorizer(min_df=5)

x_train_vector = vectorizer.fit_transform(train.FullDescription)
x_test_vector = vectorizer.transform(test.FullDescription)

'''
Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. 
Код для этого был приведен выше.
'''
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

'''
Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
'''
enc = DictVectorizer()
x_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

'''
Объедините все полученные признаки в одну матрицу "объекты-признаки". 
Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. 
Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack
'''
x_train = hstack([x_train_vector, x_train_categ])
x_test = hstack([x_test_vector, x_test_categ])
'''
Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. 
Целевая переменная записана в столбце SalaryNormalized.
'''
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(x_train, train.SalaryNormalized)

'''
Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
Значения полученных прогнозов являются ответом на задание. 
Укажите их через пробел.
'''
res = clf.predict(x_test)
print(res)

file = open('s1.txt', 'w')
file.write(' '.join(map(str, res.round(2))))
file.close()


print('fin')