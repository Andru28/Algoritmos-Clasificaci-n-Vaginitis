# Algoritmos Clasificación Vaginitis

Dataset y Algoritmos de Clasificación

## Dataset

A continuación, se encuentran los diez dataset obtenidos de los 600 casos clinicos con los que se cuenta siguiendo el enfoque OVO-OVA. El dataset1 se encuentra conformado por Gardnerella y Candidiasis, el dataset2 con Gardnerella y Tricomoniasis, el dataset3 con Gardnerella y Chlamydia, el dataset4 con Candidiasis y Tricomoniasis, el dataset5 Candidiasis y Chlamydia, el dataset6 con Tricomonisis y Chlamydia, el dataset7 con Gardnerella y Candidiasis-Tricomoniasis-Chlamydia, el dataset8 con Gardnerella-Tricomoniasis-Chlamydia y Candidiasis, el dataset9 con Gardnerella-Candidiasis-Tricomoniasis y Chlamydia y finalmente el dataset10 con Gardnerella-Candidiasis-Chlamydia y Tricomoniasis.

## Algoritmos

Con el fin de obtener un estudio comparativo en el cual se probaran diversos algoritmos para encontrar el que mejor precisión tenga en la predicción del diagnóstico de Vaginitis, se seleccionaron 5 algoritmos que seran evaluados mediante la matriz de confusión para describir el rendimiento de cada uno de ellos.

## Perceptron Simple

El perceptrón simple es un modelo neuronal unidireccional, compuesto por dos capas de neuronas, una de entrada y otra de salida. Las neuronas de entrada no realizan ningún cómputo, únicamente envían la información (en principio consideremos señales discretas {0,1}) a las neuronas de salida.

## Naïve Bayes

Es un método basado en la teoría de la probabilidad, usa frecuencias para calcular probabilidades condicionales para calcular predicciones sobre nuevos casos. Naïve Bayes es una técnica tanto predictiva como descriptiva. A pesar de ser simple, ha sido desarrollada con éxito, produciendo buenos resultados en sus aplicaciones.

## CART

El método CART produce árboles de decisión que son estrictamente binarios, que contienen ramas exactas para cada nodo de decisión. CART divide de forma recursiva los registros del conjunto de datos de entrenamiento en subconjuntos de registros con valores similares para el atributo objetivo.

## Random Forest

También conocido como “Bosques aleatorios” es un método versátil de aprendizaje automático capaz de realizar tanto tareas de regresión como de clasificación. También lleva a cabo métodos de reducción dimensional, trata valores perdidos, valores atípicos y otros pasos esenciales de exploración de datos.

## Adaboost

El algoritmo AdaBoost (refuerzo adaptativo) fue propuesto por Yoav Freund y Robert Shapire en 1995 para generar un clasificador fuerte a partir de un conjunto de clasificadores débiles. AdaBoost crea un clasificador fuerte combinando múltiples clasificadores de bajo rendimiento para que obtenga un clasificador fuerte de alta precisión.

## Referencias

Wu, S., & Nagahashi, H. (2015). Analysis of generalization ability for different AdaBoost variants based on classification and regression trees. Journal of Electrical and Computer Engineering, 2015.

Rutkowski, L., Jaworski, M., Pietruczuk, L., & Duda, P. (2014). The CART decision tree for mining data streams. Information Sciences, 266, 1-15.

Sarica, A., Cerasa, A., & Quattrone, A. (2017). Random forest algorithm for the classification of neuroimaging data in Alzheimer's disease: A systematic review. Frontiers in aging neuroscience, 9, 329.

Calvo,   D.   (2018).Perceptrón   –   red   neuronal. Tomado   dehttp://www.diegocalvo.es/perceptron/ (08/10/2018).

Torres,  L.  C.  (2011).    El  perceptrón,  redes  neuronales  artificiales.    Tomado  de https://disi.unal.edu.co/lctorress/RedNeu/LiRna004.pdf (01/03/2011).

Pacheco Leal, S. D., D. O. L. G. . G. F.-R. (2005).  El clasificador Naïve Bayes en la extracción de conocimiento de bases de datos. Ingenierías, 24–33.
