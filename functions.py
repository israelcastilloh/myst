
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: PROYECTO FINAL DE LA CLASE DE TRADING                                                      -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch
import pandas as pd
from visualizations import *
from data import *

# -- ASPECTOS ESTADISTICOS ------------------------------------------------------------------------------------------ #
# -- ---------------------------------------------------------------------------------------------------------------- #
# Datos necesarios
save_pkl('USD_MXN')
datos_divisa = read_pkl('USD_MXN')
test = datos_divisa[:'01-01-2019']
train = datos_divisa['01-01-2019':]

# -- ---------------------------------------------------------------------------------------------------------------- #
# Verficar estacionaridad de una serie de datos


def check_stationarity(param_data):
    """
    Funcion que verifica si la serie es estacionaria o no, en caso de que no lo sea se realizan diferenciaciones.
    Solo se permiten 3 diferenciaciones para no sobrediferenciar la serie.
    Parameters
    ---------
    param_data: DataFrame: Df que contiene los datos de precios de la divisa seleccionada
    Returns
    ---------
    stationary_series: list: lista que contiene la serie estacionaria de precios calculada.
    lags: int: numero de rezagos necesarios para que la serie sea estacionaria.
    Debuggin
    ---------
    stationary_series, lags = check_stationarity(datos_divisa)
    """
    # Seleccionar la columna del Dataframe que se utilizará
    param_data = param_data['Open']
    # Utilizar prueba dicky fuller para saber si es estacionaria
    test_results = sm.tsa.stattools.adfuller(param_data)
    # Cuando se cumple la siguiente condicion se considera que la serie es estacionaria
    if test_results[0] < 0 and test_results[1] <= 0.05:
        stationary_s = param_data
        lags = 0
    # Cuando no se cumple se debe diferenciar para volver la serie estacionaria
    # Solo se permiten tres rezagos para no sobrediferenciar la serie
    else:
        for i in range(1, 4):
            # Diferenciar datos
            new_data = np.diff(param_data)
            # Volver a calcular test dicky fuller
            new_results = sm.tsa.stattools.adfuller(new_data)
            # Volver a comparar para decidir si es o no estacionaria
            if new_results[0] < 0 and new_results[1] <= 0.05:
                # Nueva serie estacionaria
                # rezagos necesarios para volverlo estacionario
                stationary_s = new_data
                lags = i
                break
            else:
                param_data = new_data
    return stationary_s, lags


stationary_series, lags = check_stationarity(train)
if lags != 0:
    estacionaridad = 'No'
else:
    estacionaridad = 'Sí'
# -- ---------------------------------------------------------------------------------------------------------------- #
# Componente de autocorrelacion y autocorrelacion parcial


def significant_lag(all_coef):
    """
    Funcion que busca los rezagos más significativos de la fac y fac parcial, solo en los primeros 7.
    Parameters
    ---------
    all_coef: DataFrame: DataFrame que contiene los valores significativos de la fac y fac parcial
    Returns
    ---------
    answer: int: posición del rezago más significativo
    Debuggin
    ---------
    p = significant_lag(p_s)
    """
    # Tomar los indices de los rezagos
    ind_c = all_coef.index.values
    # Solo buscar en los primeros siete rezagos
    sig_i = ind_c[ind_c < 7]
    # Nuevos coeficientes con los seleccionados
    new_coef = all_coef[all_coef.index.isin(list(sig_i))]
    if len(new_coef) > 1:
        # Tomar los valores absolutos
        abs_coef = new_coef[1:].abs()
        # Buscar el maximo
        max_coef = abs_coef.max()
        # El indice es el rezago al que pertenece
        answer = abs_coef[abs_coef == max_coef[0]].dropna().index[0]
        return answer
    else:
        return 1


def f_autocorrelation(param_data):
    """
    Funcion que calcula el valor de p y q, parametros correspondientes a un arima.
    Parameters
    ---------
    param_data: DataFrame: DataFrame que contiene la serie de datos de la divisa seleccionada.
    Returns
    ---------
    p: int: parametro de p
    q: int: parametro de q
    Debuggin
    ---------
    p, q = f_autocorrelation(datos_divisa)
    """
    param_data = param_data['Open']
    # lambda para tomar los coef significativos
    all_significant_coef = lambda x: x if abs(x) > 0.5 else None
    # Calcular coeficientes de fac parcial
    facp = sm.tsa.stattools.pacf(param_data)
    # Pasar lambda y quitar los que no son significativos
    p_s = pd.DataFrame(all_significant_coef(facp[i]) for i in range(len(facp))).dropna()
    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
    p = significant_lag(p_s)
    # --- #
    # Calcular coeficientes de fac
    fac = sm.tsa.stattools.acf(param_data, fft=False)
    # Pasar lambda y quitar los que no son significativos
    q_s = pd.DataFrame(all_significant_coef(fac[i]) for i in range(len(fac))).dropna()
    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
    q = significant_lag(q_s)
    return p, q


p, q = f_autocorrelation(train)
if p and q != 0:
    autocorrelacion = 'Existen rezagos significativos'
else:
    autocorrelacion = 'No existen rezagos significativos'

# -- ---------------------------------------------------------------------------------------------------------------- #
# Prueba de normalidad de los datos


def check_noramlity(param_data):
    """
    Funcion que verifica si los datos de una serie se distribuyen normalmente.
    Parameters
    ---------
    param_data: DataFrame: DataFrame que contiene la serie de datos de la divisa seleccionada.
    Returns
    ---------
    norm: boolean: indica true si los datos son normales, o false si no lo son
    Debuggin
    ---------
    check_noramlity(datos_divisa)
    """
    param_data = param_data['Open']
    # shapiro test
    # Hipotesis nula: los datos se distribuyen normalmente
    # p-value < alpha se rechaza
    # p-value > alpha se acepta la hipótesis nula
    normalidad = shapiro(param_data)
    alpha = .05  # intervalo de 95% de confianza
    # si el p-value es menor a alpha, rechazamos la hipotesis de normalidad
    norm = True if normalidad[1] > alpha else False
    return norm


norm = check_noramlity(train)
if norm:
    normalidad = 'Los datos son normales'
else:
    normalidad = 'Los datos no son normales'


# -- ---------------------------------------------------------------------------------------------------------------- #
# Prueba de estacionalidad de la serie de tiempo


def check_seasonal(a=False):
    if a:
        f, pxx_den = get_periodogram(True)
        top_50_periods = {}
        # get indices for 3 highest Pxx values
        top50_freq_indices = np.flip(np.argsort(pxx_den), 0)[3:50]

        freqs = f[top50_freq_indices]
        power = pxx_den[top50_freq_indices]
        periods = 1 / np.array(freqs)
        matrix = pd.DataFrame(columns=["power", "periods"])
        matrix.power = power
        matrix.periods = periods
        print(matrix)

# -- ---------------------------------------------------------------------------------------------------------------- #
# Deteccion de datos atipicos

# -- ---------------------------------------------------------------------------------------------------------------- #
# Prueba de heterodasticidad de los residuos


def check_hetero(param_data):
    """
    Funcion que verifica si los residuos de una estimacion son heterosedasticos
    Parameters
    ---------
    param_data: DataFrame: DataFrame que contiene la serie de datos de la divisa seleccionada.
    Returns
    ---------
    norm: boolean: indica true si los datos presentan heterodasticidad, o false si no la presentan.
    Debuggin
    ---------
    check_hetero(datos_divisa)
    """
    param_data = param_data['Open']
    # arch test
    heterosced = het_arch(param_data)
    alpha = .05  # intervalo de 95% de confianza
    # si p-value menor a alpha se concluye que no hay heterodasticidad
    heter = True if heterosced[1] > 0.05 else False
    return heter

# -- ---------------------------------------------------------------------------------------------------------------- #
# Visualizacion de datos de los aspectos estadisticos


lista1 = [estacionaridad]
lista2 = [autocorrelacion]
lista3 = [normalidad]
lista4 = ['Sí']


def get_dfestadisticos(lista1, lista2, lista3, lista4):
    tabla = pd.DataFrame(columns=['Estacionaridad', 'Autocorrelacion y Autocorrelacion parcial',
                                     'Prueba de normalidad', 'Estacionalidad'])
    tabla['Estacionaridad'] = lista1
    tabla['Autocorrelacion y Autocorrelacion parcial'] = lista2
    tabla['Prueba de normalidad'] = lista3
    tabla['Estacionalidad'] = lista4
    return tabla


df_estadisticos = get_dfestadisticos(lista1, lista2, lista3, lista4)
