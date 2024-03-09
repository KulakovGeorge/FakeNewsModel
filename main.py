import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer as TVector
from sklearn.model_selection import train_test_split as splitData
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score as ScoreModel
from sklearn.metrics import confusion_matrix as err_matrix
from sklearn.metrics import classification_report as ReportModel

data = pd.read_csv('fake_news.csv')
matrix = data['text'].apply(lambda t: t.lower()) #записи новостей(привидины к нижнему регистру)
labels = data['label'] #записи меток
labels_info = labels.value_counts()

tfvec = TVector(stop_words='english', max_df=0.9)
matrix = tfvec.fit_transform(matrix)

plt.bar(labels_info.index, labels_info)
plt.title('Количество реальных и фейковых новостей в датасете')
plt.ylabel('Count')
plt.show()

train_x, test_x, train_y, test_y = splitData(matrix, labels, test_size=0.3)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(train_x, train_y)
predict = pac.predict(test_x)

matrix_error = err_matrix(test_y, predict, labels = ['FAKE', 'REAL'])
plt.imshow(matrix_error, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.colorbar()

classes = list(labels_info.index)
ticks = np.arange(len(classes))
plt.xticks(ticks, classes)
plt.yticks(ticks, classes)
plt.ylabel('Значение')
plt.xlabel('Предположение')

thresh = matrix_error.max() / 2.
for i in range(matrix_error.shape[0]):
    for j in range(matrix_error.shape[1]):
        plt.text(j, i, format(matrix_error[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matrix_error[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

report = ReportModel(test_y, predict)
print('Отчет о классификаторе: \n', report)
score = ScoreModel(test_y, predict)
print(f'Точность модели на тестовой выборке: {score}')






