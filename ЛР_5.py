import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Загрузка данных
passengers = pd.read_csv('passengers.csv', parse_dates=['Month'], index_col='Month')
births = pd.read_csv('births.csv', parse_dates=['Date'], index_col='Date')

# Функция для расчета автокорреляционной функции вручную
def calculate_acf(series, lags):
    acf_values = []
    for lag in range(1, lags + 1):
        shifted = series.shift(lag)
        corr = series.corr(shifted)
        acf_values.append(corr)
    return acf_values

# Задание 1: Посчитать автокорреляционную функцию (acf) вручную для passengers
acf_values_passengers = calculate_acf(passengers['#Passengers'], 50)
print("Задание 1: Автокорреляционная функция (ACF) для пассажиров (вручную):")
print(acf_values_passengers)

# Построение графика для acf вручную для passengers
plt.figure(figsize=(10, 6))
plt.stem(range(1, 51), acf_values_passengers, linefmt='b-', markerfmt='bo', label='Ручной ACF')

# Задание 2: Построить график для acf с помощью statsmodels для passengers
plot_acf(passengers['#Passengers'], lags=50, ax=plt.gca(), color='r', label='Statsmodels ACF')
plt.xlabel('Лаг')
plt.ylabel('ACF')
plt.title('Автокорреляционная функция (ACF) для пассажиров')
plt.legend()
plt.show()

# Задание 3: Повторить задания 1 и 2 для датасета births
acf_values_births = calculate_acf(births['Births'], 50)
print("Задание 3: Автокорреляционная функция (ACF) для рождений (вручную):")
print(acf_values_births)

# Построение графика для acf вручную для births
plt.figure(figsize=(10, 6))
plt.stem(range(1, 51), acf_values_births, linefmt='b-', markerfmt='bo', label='Ручной ACF')

# Построение графика для acf с помощью statsmodels для births
plot_acf(births['Births'], lags=50, ax=plt.gca(), color='r', label='Statsmodels ACF')
plt.xlabel('Лаг')
plt.ylabel('ACF')
plt.title('Автокорреляционная функция (ACF) для рождений')
plt.legend()
plt.show()

# Задание 4: Построить значения функции sin x и посчитать acf
x = np.arange(0, 50.1, 0.1)
y = np.sin(x)

# Посчитать автокорреляционную функцию вручную для sin(x)
acf_values_sin = calculate_acf(pd.Series(y), 100)
print("Задание 4: Автокорреляционная функция (ACF) для sin(x) (вручную):")
print(acf_values_sin)

# Построение графика для acf вручную для sin(x)
plt.figure(figsize=(10, 6))
plt.stem(range(1, 101), acf_values_sin, linefmt='b-', markerfmt='bo', label='Ручной ACF')

# Построение графика для acf с помощью statsmodels для sin(x)
plot_acf(pd.Series(y), lags=100, ax=plt.gca(), color='r', label='Statsmodels ACF')
plt.xlabel('Лаг')
plt.ylabel('ACF')
plt.title('Автокорреляционная функция (ACF) для sin(x)')
plt.legend()
plt.show()

# Задание 5: Исследовать датасет births на стационарность
result = adfuller(births['Births'])
print("Задание 5: Исследование датасета рождений на стационарность:")
print('ADF Статистика:', result[0])
print('p-значение:', result[1])
print('Критические значения:', result[4])
