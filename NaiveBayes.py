import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataframe = pd.read_csv('dataset1.csv', sep=';')
dataframe.head()

y=dataframe.clase
X=dataframe.drop('clase',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

modelo = GaussianNB()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_train)

#expected = y_train
#predicted = modelo.predict(X_train)

#print(metrics.classification_report(y_train, y_pred))
#print(metrics.confusion_matrix(y_train, y_pred))

print('Precisión: {:.2f}'
      .format(modelo.score(X_train, y_pred)))

#matriz = confusion_matrix(y_test, y_pred)
#print('Matriz de Confusión:')
#print(matriz)

#cm = ConfusionMatrix(X_test, y_pred)
cm = ConfusionMatrix(y_train, y_pred)
cm.print_stats()
