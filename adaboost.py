from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

v_data = pd.read_csv('dataset7.csv', sep=';')
print ("Dataset Length: ", len(v_data)) 
print ("Dataset Shape: ", v_data)

X = v_data.values[:, 0:9] 
y = v_data.values[:, 10] 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print("Valores predicción: ", predictions)
print("Reporte : ", 
    classification_report(y_test, predictions))
print("Matriz de confusión: ",confusion_matrix(y_test, predictions))


