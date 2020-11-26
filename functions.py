
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
pd.options.mode.use_inf_as_na = True

# -- ASPECTOS ESTADISTICOS ------------------------------------------------------------------------------------------ #

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

# -- ---------------------------------------------------------------------------------------------------------------- #
# Prueba de estacionalidad de la serie de tiempo


def check_seasonal(data):
    f, pxx_den = get_periodogram(data)
    top_50_periods = {}
    # get indices for 3 highest Pxx values
    top50_freq_indices = np.flip(np.argsort(pxx_den), 0)[3:6]

    freqs = f[top50_freq_indices]
    power = pxx_den[top50_freq_indices]
    periods = 1 / np.array(freqs)
    matrix = pd.DataFrame(columns=["power", "periods"])
    matrix.power = power
    matrix.periods = periods
    return matrix

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
# Funcion que corre las pruebas estadisticas


def get_statistics(data):
    # Verificar estacionaridad
    stationary_series, lags = check_stationarity(data)
    if lags != 0:
        estacionaridad = 'No'
    else:
        estacionaridad = 'Sí'
    # Verificar Fac y Fac Parcial
    p, q = f_autocorrelation(data)
    if p and q != 0:
        autocorrelacion = 'Existen rezagos significativos'
    else:
        autocorrelacion = 'No existen rezagos significativos'
    # Verificar normalidad
    norm = check_noramlity(data)
    if norm:
        normalidad = 'Los datos son normales'
    else:
        normalidad = 'Los datos no son normales'
    # Verificar si la serie es ciclica
    seasonal = 'Sí'
    return estacionaridad, autocorrelacion, normalidad, seasonal

# -- ---------------------------------------------------------------------------------------------------------------- #
# Visualizacion de datos de los aspectos estadisticos


def get_dfestadisticos(valor1, valor2, valor3, valor4):
    lista1 = [valor1]
    lista2 = [valor2]
    lista3 = [valor3]
    lista4 = [valor4]
    tabla = pd.DataFrame(columns=['Estacionaridad', 'Autocorrelacion y Autocorrelacion parcial',
                                     'Prueba de normalidad', 'Estacionalidad'])
    tabla['Estacionaridad'] = lista1
    tabla['Autocorrelacion y Autocorrelacion parcial'] = lista2
    tabla['Prueba de normalidad'] = lista3
    tabla['Estacionalidad'] = lista4
    return tabla

# -- ASPECTOS COMPUTACIONALES --------------------------------------------------------------------------------------- #
# -- ---------------------------------------------------------------------------------------------------------------- #
# Technical Indicators


def CCI(data, ndays):
    '''
    Commodity Channel Index
    Parameters
    ----------
    data : pd.DataFrame with 3 colums named High, Low and Close
    ndays : int used for moving average and moving std
    Returns
    -------
    CCI : pd.Series containing the CCI
    '''
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) /
                    (0.015 * TP.rolling(ndays).std()), name='CCI')
    return CCI


def SMA(data, ndays):
    '''Simple Moving Average'''
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name='SMA')
    return SMA


def BBANDS(data, window):
    ''' Bollinger Bands '''
    MA = data.Close.rolling(window).mean()
    SD = data.Close.rolling(window).std()
    return MA + (2 * SD), MA - (2 * SD)


def RSI(data, window):
    ''' Relative Strnegth Index'''
    delta = data['Close'].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window).mean()
    roll_down1 = down.abs().ewm(span=window).mean()
    RS1 = roll_up1 / roll_down1
    return 100.0 - (100.0 / (1.0 + RS1))


def price_from_max(data, window):
    return data['Close'] / data['Close'].rolling(window).max()


def price_from_min(data, window):
    return data['Close'] / data['Close'].rolling(window).min() - 1


def price_range(data, window):
    pricerange = (data['Close'] - data['Close'].rolling(window).min()) / \
                 (data['Close'].rolling(window).max() - data['Close'].rolling(window).min())
    return pricerange


# %% Labeling: 1 for positive next day return, 0 for negative next day return
def next_day_ret(df):
    '''
    Given a DataFrame with one column named 'Close' label each row according to
    the next day's return. If it is positive, label is 1. If negative, label is 0
    Designed to label a dataset used to train a ML model for trading
    RETURNS
    next_day_ret: pd.DataFrame
    label: list
    Implementation on df_pe:
        _, label = next_day_ret(df_pe)
        df_pe['Label'] = label
    '''
    next_day_ret = df.Close.pct_change().shift(-1)
    label = []
    for i in range(len(next_day_ret)):
        if next_day_ret[i] > 0:
            label.append(1)
        else:
            label.append(0)
    return next_day_ret, label


# %%
# binary ,returns and accum returns

def ret_div(df):
    '''
    Return a logarithm and arithmetic daily returns
    and daily acum daily
    '''
    ret_ar = df.Close.pct_change().fillna(0)
    ret_ar_acum = ret_ar.cumsum()
    ret_log = np.log(1 + ret_ar_acum).diff()
    ret_log_acum = ret_log.cumsum()

    binary = ret_ar
    binary[binary < 0] = 0
    binary[binary > 0] = 1
    return ret_ar, ret_ar_acum, ret_log, ret_log_acum, binary

# zscore normalization


def z_score(df):
    # zscore
    mean, std = df.Close.mean(), df.Close.std()
    zscore = (df.Close - mean) / std

    return zscore


# diff integer
def int_diff(df, window: np.arange):
    diff = [
        df.Close.diff(x) for x in window
    ]
    return diff


# moving averages
def mov_averages(df, space: np.arange):
    mov_av = [
        df.Close.rolling(w).mean() for w in space
    ]
    return mov_av


def quartiles(df, n_bins: int):
    'Assign quartiles to data, depending of position'
    bin_fxn = pd.qcut(df.Close, q=n_bins, labels=range(1, n_bins + 1))
    return bin_fxn


def add_all_features(datos_divisa):
    # Technical Indicators
    datos_divisa['CCI'] = CCI(datos_divisa, 14)  # Add CCI
    datos_divisa['SMA_5'] = SMA(datos_divisa, 5)
    datos_divisa['SMA_10'] = SMA(datos_divisa, 10)
    datos_divisa['MACD'] = datos_divisa['SMA_10'] - datos_divisa['SMA_5']
    datos_divisa['Upper_BB'], datos_divisa['Lower_BB'] = BBANDS(datos_divisa, 10)
    datos_divisa['Range_BB'] = (datos_divisa['Close'] - datos_divisa['Lower_BB']) / (datos_divisa['Upper_BB'] - datos_divisa['Lower_BB'])
    datos_divisa['RSI'] = RSI(datos_divisa, 10)
    datos_divisa['Max_range'] = price_from_max(datos_divisa, 20)
    datos_divisa['Min_range'] = price_from_min(datos_divisa, 20)
    datos_divisa['Price_Range'] = price_range(datos_divisa, 50)
    datos_divisa['returna'], datos_divisa['returna_acums'], datos_divisa['returnlog'], datos_divisa['returnlog_acum'], datos_divisa['binary'] = ret_div(datos_divisa)
    datos_divisa['zscore'] = z_score(datos_divisa)
    datos_divisa['diff1'], datos_divisa['diff2'], datos_divisa['diff3'], datos_divisa['diff4'], datos_divisa['diff5'] = int_diff(datos_divisa, np.arange(1,6))
    datos_divisa['mova1'], datos_divisa['movaf2'], datos_divisa['mova3'], datos_divisa['mova4'], datos_divisa['mova5'] = mov_averages(datos_divisa,np.arange(1,6))
    datos_divisa['quartiles'] = quartiles(datos_divisa, 10)
    datos_divisa['Label'] = next_day_ret(datos_divisa)[1]
    return datos_divisa


def math_transformations(df):
    original_columns = df.columns
    for col in original_columns:
        df['sin_'+col] = np.sin(df[col].values.astype(float))
        df['cos_'+col] = np.cos(df[col].values.astype(float))
        df['square_'+col] = np.square(df[col].values.astype(float))
        df['sqrt_'+col] = np.sqrt(df[col].values.astype(float))
        df['exp_'+col] = np.exp(df[col].values.astype(float))
        df['exp2_'+col] = np.exp2(df[col].values.astype(float))
        df['tanh_'+col] = np.tanh(df[col].values.astype(float))
        df['arctan_'+col] = np.arctan(df[col].values.astype(float))
        df['log_'+col] = np.log(df[col].values.astype(float))
        df['log2_'+col] = np.log2(df[col].values.astype(float))
        df['log10_'+col] = np.log10(df[col].values.astype(float))
        df['sindiff_' + col] = np.sin(df[col].values.astype(float))**(1/2)
        df['cosdiff_' + col] = np.cos(df[col].values.astype(float))**(1/2)
        df['squarediff_' + col] = np.square(df[col].values.astype(float))**(1/2)
        df['sqrtdiff_' + col] = np.sqrt(df[col].values.astype(float))**(1/2)
        df['tanhdiff_' + col] = np.tanh(df[col].values.astype(float))**(1/2)
        df['arctandiff_' + col] = np.arctan(df[col].values.astype(float))**(1/2)
        df['logdiff_' + col] = np.log(df[col].values.astype(float))**(1/2)
        df['log2diff_' + col] = np.log2(df[col].values.astype(float))**(1/2)
        df['log10diff_' + col] = np.log10(df[col].values.astype(float))**(1/2)
    return df