# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:05:23 2024

@author: KGP
"""

# Regresión polinómica

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt #pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd #librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # Si pusiera [:, 1] en size de la tabla me daría (10,) porque lo consideraría un vector
y = dataset.iloc[:, 2].values
# iloc sirve para localizar por posición las variables, en este caso independientes
# hemos indicado entre los cochetes, coge todas las filas [:(todas las filas), :-1(todas las columnas excepto la última]
# .values significa que quiero sacar solo los valores del dataframe no las posiciones


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
# from sklearn.model_selection import train_test_split""" #En esta ocasión al haber sólo 10 datos no hacemos trainning.
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) # random_state podría coger cualquier número, es el número para poder reproducir el algoritmo

# Escalado de variables. Siguiente código COMENTADO porque se usa mucho pero no siempre
"""from sklearn.preprocessing import StandardScaler # Utilizarlo para saber que valores debe escalar apropiadamente y luego hacer el cambio
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #hacemos transform sin "fit" para que haga la transformación con los datos del transform de entrenamiento"""

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Si dejo el () vacio da grado 2 por defecto.
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualización de los resultados del modelo lineal
plt.scatter(x, y, color = "red")
plt.plot(x, lin_reg.predict(x), color = "blue")
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

# Visualización de los resultados del modelo polinómico
x_grid = np.arange(min(x), max(x), 0.1) # Me ha creado un vector fila de 1 fila y 90 columnas
x_grid = x_grid.reshape(len(x_grid), 1) # convertimos a vector columna de 1 columna y 90 filas
plt.scatter(x, y, color = "red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = "blue") # También podría usar: plt.plot(x, lin_reg_2.predict(x_poly), color = "blue")
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

# Predicción de nuestro modelos (predicción salario en punto 6.5)
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
