import pandas
import numpy as np


'''
Уменьшение количества цветов изображения
'''

'''
Загрузите картинку parrots.jpg. 
Преобразуйте изображение, приведя все значения в интервал от 0 до 1. 
Для этого можно воспользоваться функцией img_as_float из модуля skimage. 
Обратите внимание на этот шаг, так как при работе с исходным изображением вы получите некорректный результат.
'''
from skimage.io import imread
from skimage import img_as_float
image = imread('parrots.jpg')

image = img_as_float(image)

#print(image)

'''
Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - 
значениями интенсивности в пространстве RGB.

Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. 

После выделения кластеров все пиксели, отнесенные в один кластер, 
попробуйте заполнить двумя способами: медианным и средним цветом по кластеру.

Измерьте качество получившейся сегментации с помощью метрики PSNR. 

Эту метрику нужно реализовать самостоятельно (см. определение).

Найдите минимальное количество кластеров, при котором значение PSNR выше 20 
(можно рассмотреть не более 20 кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера). 
Это число и будет ответом в данной задаче.
'''

from skimage.measure import compare_psnr
from sklearn.cluster import KMeans
X = np.array(image.reshape((image.shape[0]*image.shape[1],3)))
for s in range(8, 21):
  kmeans = KMeans(init='k-means++',random_state=241,n_clusters=s)
  kmeans.fit(X)
  y = kmeans.labels_
  X_median = np.array(X)
  X_mean = np.array(X)
  for i in range(s):
    X_median[y==i] = np.median(X_median[y==i], axis=0)
    X_mean[y==i] = np.mean(X_mean[y==i], axis=0)
  image_median = X_median.reshape(image.shape[0], image.shape[1], 3)
  image_mean = X_mean.reshape(image.shape[0], image.shape[1], 3)
  print('n_clusters=', s, " PSNR_median=", compare_psnr(image, image_median), " PSNR_mean=", compare_psnr(image, image_mean))

'''
import matplotlib.pyplot as plt
plt.imshow(imfl)
plt.show()
'''
file = open('s1.txt', 'w')
file.write('11')
file.close()

print('fin')