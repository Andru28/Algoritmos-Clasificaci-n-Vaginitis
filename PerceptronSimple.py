import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
 
#Cargar datos
import pandas as pd
df = pd.read_csv('dataset3.csv', sep=';')
print('_'*60 + 'Columnas')
print(df.columns.values)
print('_'*60 + 'Información')
print (df.info())
print('_'*60 + 'Descripción')
print (df.describe().transpose())
print('_'*60 + 'SHAPE')
print (df.shape)
print('_'*60 + 'Clases')
print (df.loc[:,'clase'].value_counts())
print('_'*60 + 'Valores nulos')
print (df.isnull().sum())
print('_'*60 + 'Valores nulos bis')
print(df.isnull().values.any())

X = df.iloc[:, 0:10].values
y = df.iloc[:, 10].values
plt.title("Perceptron Simple", fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, alpha=0.5, edgecolor='c')
plt.show()

#Dividimos el dataframe en train y test
X, y = df.loc[:, ['flujo', 'secrecion', 'prurito', 'dolor abdominal', 'dolor pelvis', 'disuria', 'olor', 'poliuria', 'polaquiuria', 'dispareunia']].values, df.loc[:,['clase']].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Creamos la clase para el perceptron
class SimplePerceptron():

    def __init__(self, eta):
        """
        :param eta: tasa de aprendizaje
        """
        self.eta = eta

    def zeta(self, X):
        """
        Calcula el producto de las entradas por sus pesos
        :param X: datos de entrenamiento con las caracteristicas. Array
        """
        zeta = np.dot(1, self.weights[0]) + np.dot(X, self.weights[1:])
        return zeta

    def predict(self, X):
        """
        Calcula la salida de la neurona teniendo en cuenta la función de activación
        :param X: datos con los que predecir la salida de la neurona. Array
        :return: salida de la neurona
        """
        output = np.where(self.zeta(X) >= 0.0, 1, 0)
        return output

    def fit(self, X, y):
        #Ponemos a cero los pesos
        self.weights = [0] * (X.shape[1] + 1)
        self.errors = []
        self.iteraciones = 0
        while True:
            errors = 0
            for features, expected in zip(X,y):
                delta_weight = self.eta * (expected - self.predict(features))
                self.weights[1:] += delta_weight * features
                self.weights[0] += delta_weight * 1
                errors += int(delta_weight != 0.0)
            self.errors.append(errors)
            self.iteraciones += 1
            if errors == 0:
                break

#Creamos una instancia de la clase
sp = SimplePerceptron(eta=0.1)
#Entrenamos
sp.fit(X_train, y_train)
#Ploteamos las iteraciones y numero de errores
#plt.plot(range(1, len(sp.errors) + 1), sp.errors, marker='o')
#plt.xlabel('Iteracion')
#plt.ylabel('Errores')
#plt.tight_layout()
#plt.show()

#Comprobamos la precisión del perceptron con los datos de test
print('_'*60 + "Prediccion para X_test")
prediction = sp.predict(X_test)
print (prediction)
print('_'*60 + "Esperado para X_test")
print (y_test.T[0])
X_train=y_test.T[0]
print('_'*60 + "¿Coincide lo esperado y lo devuelto por el perceptron?")
print (np.array_equal(prediction, y_test.T[0]))
print('_'*60 + "PRECISION")
print(str(np.sum(prediction == y_test.T[0])/prediction.shape[0] * 100) + ' %')

print(confusion_matrix(X_train, prediction))
#data = {'prediction','X_train'}

#df = pd.DataFrame(data, columns=['X_train','prediction'])

#cm = ConfusionMatrix(X_train, prediction)
#cm = cm(X_train, prediction)
#cm.plot(normalized=True)
#plt.show()
#cm.print_stats()
