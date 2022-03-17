import pandas
import numpy as np
import sklearn.metrics as met



'''
Загрузите файл classification.csv. 
В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred).

Заполните таблицу ошибок классификации:
	                Actual Positive	Actual Negative
Predicted Positive	    TP	            FP
Predicted Negative	    FN	            TN

Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. 
Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1. 
Ответ в данном вопросе — четыре числа через пробел.
'''
cls = pandas.read_csv('classification.csv')

TP = 0
FP = 0
FN = 0
TN = 0

for i in range(len(cls)):
    if cls['true'][i]:
        if cls['pred'][i]:
            TP += 1
        else:
            FN += 1
    else:
        if cls['pred'][i]:
            FP += 1
        else:
            TN += 1
print(TP, FP)
print(FN, TN)

file = open('s4.txt', 'w')
file.write(str(TP) + ' ' + str(FP) + ' ' + str(FN) + ' ' + str(TN))
file.close()

'''
Посчитайте основные метрики качества классификатора:
Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
Precision (точность) — sklearn.metrics.precision_score
Recall (полнота) — sklearn.metrics.recall_score
F-мера — sklearn.metrics.f1_score

В качестве ответа укажите эти четыре числа через пробел.
'''
acc_score = met.accuracy_score(cls['true'], cls['pred'])
pre_score = met.precision_score(cls['true'], cls['pred'])
rec_score = met.recall_score(cls['true'], cls['pred'])
f_score = met.f1_score(cls['true'], cls['pred'])

print(acc_score, pre_score, rec_score, f_score)

res = []
res.append(round(acc_score, 2))
res.append(round(pre_score, 2))
res.append(round(rec_score, 2))
res.append(round(f_score, 2))

file = open('s5.txt', 'w')
file.write(' '.join(map(str, res)))
file.close()

'''
Имеется четыре обученных классификатора. 
В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
    для логистической регрессии — вероятность положительного класса (колонка score_logreg),
    для SVM — отступ от разделяющей поверхности (колонка score_svm),
    для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
    для решающего дерева — доля положительных объектов в листе (колонка score_tree).

Загрузите этот файл
'''
scr = pandas.read_csv('scores.csv')

'''
Посчитайте площадь под ROC-кривой для каждого классификатора. 
Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? 
Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
'''

l_roc = met.roc_auc_score(scr['true'], scr['score_logreg'])
svm_roc = met.roc_auc_score(scr['true'], scr['score_svm'])
met_roc = met.roc_auc_score(scr['true'], scr['score_knn'])
tree_roc = met.roc_auc_score(scr['true'], scr['score_tree'])

print(l_roc, svm_roc, met_roc, tree_roc)

file = open('s6.txt', 'w')
file.write('score_logreg')
file.close()

'''
Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?

Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. 
Она возвращает три массива: precision, recall, thresholds. 
В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. 
Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
'''
for algorythm in scr.columns[1:]:
    yscores = scr[algorythm]
    precision, recall, thresholds = met.precision_recall_curve(scr['true'], yscores)

    d = {'precision' : precision, 'recall' : recall}
    df = pandas.DataFrame(d)
    print(df[df['recall']>=0.7].max()['precision'], yscores.name)

file = open('s7.txt', 'w')
file.write('score_tree')
file.close()


print('fin')