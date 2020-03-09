import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from datetime import datetime
start_time = datetime.now()


# CARGAR DATASET DE DROPBOX
#---------------------------------------------------------------------------------------------
data = pd.read_csv('dataset10.csv', sep=';')
clase_name = 'clase' # nombre de variable a predecir
headers    = data.columns.values.tolist()
headers.remove(clase_name)

# TRAIN y TEST
#---------------------------------------------------------------------------------------------
X = data.iloc[:, 0:10].values
y = data.iloc[:, 10].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# MODELO
#---------------------------------------------------------------------------------------------
modelo = RandomForestClassifier(
 random_state      = 1,   # semilla inicial de aleatoriedad del algoritmo
 n_estimators      = 100, # cantidad de arboles a crear
 min_samples_split = 2,   # cantidad minima de observaciones para dividir un nodo
 min_samples_leaf  = 1,   # observaciones minimas que puede tener una hoja del arbol
 n_jobs            = 1    # tareas en paralelo. para todos los cores disponibles usar -1
 )
modelo.fit(X = X_train, y = y_train)


# PREDICCION
#---------------------------------------------------------------------------------------------
prediccion = modelo.predict(X_test)


# METRICAS
#---------------------------------------------------------------------------------------------
print(metrics.classification_report(y_true=y_test, y_pred=prediccion))
print(pd.crosstab(y_test, prediccion, rownames=['REAL'], colnames=['PREDICCION']))


# IMPORTANCIA VARIABLES
#---------------------------------------------------------------------------------------------
var_imp = pd.DataFrame({
 'Característica':headers, 
 'Importancia':modelo.feature_importances_.tolist()
 })
print(var_imp.sort_values(by = 'Importancia', ascending=False))


# END
#---------------------------------------------------------------------------------------------
end_time = datetime.now()
print('duración: ' + format(end_time - start_time))

