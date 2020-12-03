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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import svm
from gplearn.genetic import SymbolicTransformer
from statsmodels.stats.diagnostic import het_arch
import pandas as pd
from sklearn.model_selection import train_test_split
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
    """
    Funcion que verifica si los datos de una serie presentan cierta estacionalidad.
    Parameters
    ---------
    data: DataFrame: DataFrame que contiene la serie de datos de la divisa seleccionada.
    Returns
    ---------
    matrix: DataFrame: df que contiene los datos de los periodos en los que se encuentra cierta repeticion.
    Debuggin
    ---------
    ciclos = ft.check_seasonal(train)
    """
    f, pxx_den = signal.periodogram(data['Open'], 1)
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
    # arch test
    heterosced = het_arch(param_data)
    alpha = .05  # intervalo de 95% de confianza
    # si p-value menor a alpha se concluye que no hay heterodasticidad
    heter = True if heterosced[1] > alpha else False
    return heter


# -- ---------------------------------------------------------------------------------------------------------------- #
# Funcion que corre las pruebas estadisticas


def get_statistics(data):
    """
    Funcion que calcula todas las pruebas estadisticas y otorga cierto valor cualitativo a los resultados
    Parameters
    ---------
    data: DataFrame: DataFrame que contiene la serie de datos de la divisa seleccionada.
    Returns
    ---------
    estacionaridad: str: indica si hay o no estacionairdad en la serie
    autocorrelacion: str: indica si hay o no rezagos significativos en la serie
    normalidad: str: indica si hay o no normalidad en la serie
    seasonal: str: indica si hay o no estacionalidad en la serie
    atipicos: str: indica si hay datos atipicos o no en la serie
    Debuggin
    ---------
    estacionaridad, autocorrelacion, normalidad, seasonal, atipicos = ft.get_statistics(train)
    """
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
    atipicos = 'No'
    return estacionaridad, autocorrelacion, normalidad, seasonal, atipicos


# -- ---------------------------------------------------------------------------------------------------------------- #
# Visualizacion de datos de los aspectos estadisticos


def get_dfestadisticos(valor1, valor2, valor3, valor4, valor5):
    """
    Funcion que crea un dataframe de los resultados estadisticos
    Parameters
    ---------
    valor1: str: indica si hay o no estacionairdad en la serie
    valor2: str: indica si hay o no rezagos significativos en la serie
    valor3: str: indica si hay o no normalidad en la serie
    valor4: str: indica si hay o no estacionalidad en la serie
    valor5: str: indica si hay datos atipicos o no en la serie
    Returns
    ---------
    tabla: Dataframe: df que contiene los valores cualitativos de resultados estadisticos.
    Debuggin
    ---------
    check_hetero(datos_divisa)
    """
    lista1 = [valor1]
    lista2 = [valor2]
    lista3 = [valor3]
    lista4 = [valor4]
    lista5 = [valor5]
    tabla = pd.DataFrame(columns=['Estacionaridad', 'Autocorrelacion y Autocorrelacion parcial',
                                  'Prueba de normalidad', 'Estacionalidad', 'Valores Atipicos'])
    tabla['Estacionaridad'] = lista1
    tabla['Autocorrelacion y Autocorrelacion parcial'] = lista2
    tabla['Prueba de normalidad'] = lista3
    tabla['Estacionalidad'] = lista4
    tabla['Valores Atipicos'] = lista5
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

# -------------------------------- MODEL: Multivariate Linear Regression Models with L1L2 regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def mult_reg(p_x, p_y):
    """
    Funcion para ajustar varios modelos lineales

    Parameters
    ----------

    p_x: pd.DataFrame with regressors or predictor variables

    p_y: pd.DataFrame with variable to predict

    Returns
    -------
    r_models: dict Diccionario con modelos ajustados

    """
    xtrain, xtest, ytrain, ytest = train_test_split(p_x, p_y, test_size=.8, random_state=455)

    # fit linear regression
    linreg = LinearRegression(normalize=False, fit_intercept=False)
    linreg.fit(xtrain, ytrain)
    y_p_linear = linreg.predict(xtest)

    # Fit RIDGE regression
    ridgereg = Ridge(normalize=True)
    model = ridgereg.fit(xtrain, ytrain)
    y_p_ridge = model.predict(xtest)

    # Fit LASSO regression
    lassoreg = Lasso(normalize=True)
    lassoreg.fit(xtrain, ytrain)
    y_p_lasso = lassoreg.predict(xtest)

    # Fit ElasticNet regression
    enetreg = ElasticNet(normalize=True)
    enetreg.fit(xtrain, ytrain)
    y_p_enet = enetreg.predict(xtest)

    # RSS = residual sum of squares

    # Return the result of the model
    r_models = {"summary": {"linear rss": sum((y_p_linear - ytest) ** 2),
                            "Ridge rss": sum((y_p_ridge - ytest) ** 2),
                            "lasso rss": sum((y_p_lasso - ytest) ** 2),
                            "elasticnet rss": sum((y_p_enet - ytest) ** 2)},
                "test": ytest,
                'linear': {'rss': sum((y_p_linear - ytest) ** 2),
                           'predict': y_p_linear,
                           'model': linreg,
                           'intercept': linreg.intercept_,
                           'coef': linreg.coef_},
                'ridge': {'rss': sum((y_p_ridge - ytest) ** 2),
                          'predict': y_p_ridge,
                          'model': ridgereg,
                          'intercept': ridgereg.intercept_,
                          'coef': ridgereg.coef_},
                'lasso': {'rss': sum((y_p_lasso - ytest) ** 2),
                          'predict': y_p_lasso,
                          'model': lassoreg,
                          'intercept': lassoreg.intercept_,
                          'coef': lassoreg.coef_},
                'elasticnet': {'rss': sum((y_p_enet - ytest) ** 2),
                               'predict': y_p_enet,
                               'model': enetreg,
                               'intercept': enetreg.intercept_,
                               'coef': enetreg.coef_}
                }

    return r_models


# ------------------------------------------------------------------ MODEL: Symbolic Features Generation -- #
# --------------------------------------------------------------------------------------------------------- #


def symbolic_features(p_x, p_y):
    """
    Funcion para crear regresores no lineales

    Parameters
    ----------
    p_x: pd.DataFrame
        with regressors or predictor variables
        p_x = data_features.iloc[0:30, 3:]

    p_y: pd.DataFrame with variable to predict

    Returns
    -------
    results: model

    """
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log', "max", "min", "sin",
                                              "cos"],
                                population_size=5000, hall_of_fame=100, n_components=20,
                                generations=20, tournament_size=20, stopping_criteria=.05,
                                const_range=None, init_depth=(4, 12),
                                metric='pearson', parsimony_coefficient=0.001,
                                p_crossover=0.4, p_subtree_mutation=0.2, p_hoist_mutation=0.1,
                                p_point_mutation=0.3, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
                                warm_start=True)

    init = model.fit_transform(p_x[:'01-01-2019'], p_y[:'01-01-2019'])
    model_params = model.get_params()
    gp_features = model.transform(p_x)
    model_fit = np.hstack((p_x, gp_features))
    results = {'fit': model_fit, 'params': model_params, 'model': model}

    return results


def f_features(p_data, p_nmax):
    # reasignar datos
    data = p_data.copy()
    data["predict"] = data['Close'].shift(-1)
    # pips descontados al cierre
    data['co'] = (data['Close'] - data['Open']) * 10000
    # pips descontados alcistas
    data['ho'] = (data['High'] - data['Open']) * 10000
    # pips descontados bajistas
    data['ol'] = (data['Open'] - data['Low']) * 10000
    # pips descontados en total (medida de volatilidad)
    data['hl'] = (data['High'] - data['Low']) * 10000

    for n in range(0, p_nmax):

        # rezago n de Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
        # rezago n de High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
        # rezago n de High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)

        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 2)] = data['ol'].rolling(n + 2).mean()
        # promedio movil de ventana n
        data['ma_ho_' + str(n + 2)] = data['ho'].rolling(n + 2).mean()
        # promedio movil de ventana n
        data['ma_hl_' + str(n + 2)] = data['hl'].rolling(n + 2).mean()

        # hadamard product of previously generated features
        list_hadamard = [data['lag_ol_' + str(n + 1)], data['lag_ho_' + str(n + 1)],
                         data['lag_hl_' + str(n + 1)]]

        for x in list_hadamard:
            data['had_' + 'lag_oi_' + str(n + 1) + '_' + 'ma_ol_' + str(n + 2)] = x * data['ma_ol_' + str(n + 2)]
            data['had_' + 'lag_oi_' + str(n + 1) + '_' + 'ma_ho_' + str(n + 2)] = x * data['ma_ho_' + str(n + 2)]
            data['had_' + 'lag_oi_' + str(n + 1) + '_' + 'ma_hl_' + str(n + 2)] = x * data['ma_hl_' + str(n + 2)]
    picos = check_seasonal(p_data)
    t = np.arange(1, len(data) + 1)
    data["t"] = t
    # data['RSI'] = RSI(p_data, 10)
    # data['CCI'] = CCI(p_data, 14)  # Add CCI
    # data['SMA_5'] = SMA(p_data, 5)
    # data['SMA_10'] = SMA(p_data, 10)
    # data['MACD'] = data['SMA_10'] - data['SMA_5']
    # data['Upper_BB'], data['Lower_BB'] = BBANDS(p_data, 10)
    # data['Range_BB'] = (data['Close'] - data['Lower_BB']) / (data['Upper_BB'] - data['Lower_BB'])
    for i in picos.periods:
        data[f"{i:.2f}_sen"] = np.abs(np.sin(((2 * np.pi) / i) * t))
        data[f"{i:.2f}_cos"] = np.abs(np.cos(((2 * np.pi) / i) * t))
    data = data.drop(['Open', 'High', 'Low', 'Close'], axis=1)
    data = data.dropna(axis='rows')
    return data


def recursivo(variables, real, modelo):
    """
    Funcion que hace la prediccion de los datos con el modelo seleccionado
    Parameters
    ---------
    variables:
    real:
    modelo:
    Returns
    ---------
    predict_ridge:
    Debuggin
    ---------
    prediccion = ft.recursivo(nuevos_features, features_divisa, lm_model_s["ridge"]["model"])
    """
    predict_ridge = pd.DataFrame(index=variables.index[931:], columns=["predicted", "real"])
    predict_ridge["real"] = real.iloc[:, 0]['01-01-2019':]
    for period in range(0, len(variables['01-01-2019':])):
        xtrain = variables.iloc[:len(variables[:'01-01-2019'])+period, 1:]
        xtest = variables.iloc[len(variables['01-01-2019'])+period:len(variables['01-01-2019'])+period+1, 1:]
        ytrain = real.iloc[:len(real[:'01-01-2019'])+period, 0]
        model = modelo.fit(xtrain, ytrain)
        y_p_ridge = model.predict(xtest)
        predict_ridge.iloc[period, 0] = float(y_p_ridge)
    return predict_ridge


pd.set_option("display.max_rows", None, "display.max_columns", 10)
pd.set_option('display.float_format', '{:.2f}'.format)


def backtest(prediccion, historicos):
    """
    Funcion que realiza el backtest del modelo aplicado a trading
    Parameters
    ---------
    prediccion: DataFrame: df que contiene la prediccion con el modelo seleccionado
    historicos: DataFrame: df que contiene los datos historicos de la divisa
    Returns
    ---------
    Backtest_df: DataFrame: df que contiene los resultados de trading y sus decisiones
    Debuggin
    ---------
    backtest = ft.backtest(prediccion, datos_divisa)
    """
    capital_total = 100_000
    monto_operacion = capital_total*.01

    backtest_df = pd.concat([prediccion, historicos.Close], axis=1).dropna()
    backtest_df = pd.concat([backtest_df, historicos.Open], axis=1).dropna()

    backtest_df['event'] = ''
    backtest_df['p_apertura'] = 0.0000
    backtest_df['p_l'] = 0.0000

    for p in range(len(backtest_df.predicted)-1):
        if backtest_df.predicted[p] < backtest_df.Close[p]:
            ########################################################
            backtest_df.event[p] = 'sell'
            backtest_df.p_apertura[p] = backtest_df.Open[p]
            backtest_df.p_l[p] = monto_operacion*(backtest_df.p_apertura[p]-backtest_df.real[p])
        else:
            backtest_df.event[p] = 'buy'
            backtest_df.p_apertura[p] = backtest_df.Open[p]
            backtest_df.p_l[p] = monto_operacion*(backtest_df.real[p]-backtest_df.p_apertura[p])
    backtest_df['cap'] = np.cumsum(backtest_df.p_l) + capital_total
    return backtest_df


# ------------------------------------------------------------------ Metricas de atribucion al desempeño -- #
# --------------------------------------------------------------------------------------------------------- #
# Calcular residuos de nuestro modelo

def get_residuos(datos):
    """
     Funcion que calcula los residuos del modelo
     Parameters
     ---------
    datos: DataFrame: df que contiene los datos de prediccion y reales
     Returns
     ---------
    residuos: list: lista que contiene los residuos del modelo.
     Debuggin
     ---------
    residuos = ft.get_residuos(backtest)
     """
    residuos = datos['real'] - datos['predicted']
    return residuos

# --------------------------------------------------------------------------------------------------------- #
# Funciones auxiliares que calculen los datos de minimos y maximos

def get_maximo(data, ind, param=False):
    """
     Función que me encuentra máximo y su información (Indice y fecha) del DataFrame
     Parameters
     ---------
     data: DataFrame: df Data Frame que contiene la lista en la que queremos encontrar el máximo
     ind: int: Entero que señala el indice desde el cual buscamos ese máximo
     param: Función de activación para buscar a la derecha del indice
     Returns
     ---------
     maximo: float: el valor maximo encontrado
     indice_max: int: Lugar donde se encuentra el valor máximo en el DataFrame
     fecha: Datetime: Es la fecha en la cual se encontro el valor máximo
     Debuggin
     ---------
     maximo_1_d, inidice_maximo_1_d, fecha_max_1_d = get_maximo(param_data.loc[:indice_minimo_1_d], 0)
     maximo_2, inidice_maximo_2, fecha_max_2 = get_maximo(param_data.loc[indice_minimo_2:], indice_minimo_2, True)
     """
    maximo = np.max(data['cap'])
    indice_max = (np.where(data['cap'] == maximo))[0]
    if len(indice_max) > 1:
        indice_max = int(indice_max[0])
    else:
        indice_max = int(indice_max)
    if param:
        indice_max = indice_max+ind
        fecha = pd.to_datetime(data.loc[indice_max]['timestamp']).date()
    else:
        fecha = pd.to_datetime(data.loc[indice_max]['timestamp']).date()
    return maximo, indice_max, fecha


# -- ---------------------------------------------------------------------------------------------------------------- #


def get_minimo(data, ind, param=False):
    """
     Función que me encuentra mínimo y su información (Indice y fecha) del DataFrame
     Parameters
     ---------
     data: DataFrame: df Data Frame que contiene la lista en la que queremos encontrar el mínimo
     ind: int: Entero que señala el indice desde el cual buscamos ese mínimo
     param: Función de activación para buscar a la derecha del indice
     Returns
     ---------
     mínimo: float: el valor mínimo encontrado
     indice_min: int: Lugar donde se encuentra el valor mínimo en el DataFrame
     fecha: Datetime: Es la fecha en la cual se encontro el valor mínimo
     Debuggin
     ---------
     minimo_1_d, indice_minimo_1_d, fecha_min_1_d = get_minimo(param_data, 0)
     minimo_2_d, inidice_minimo_2_d, fecha_min_2_d = get_maximo(param_data.loc[indice_maximo_2_d:], indice_maximo_2_d, True)
     """
    minimo = np.min(data['cap'])
    indice_min = (np.where(data['cap'] == minimo))[0]
    if len(indice_min) > 1:
        indice_min = int(indice_min[0])
    else:
        indice_min = int(indice_min)
    if param:
        indice_min = indice_min+ind
        fecha = pd.to_datetime(data.loc[indice_min]['timestamp']).date()
    else:
        fecha = pd.to_datetime(data.loc[indice_min]['timestamp']).date()
    return minimo, indice_min, fecha


# --------------------------------------------------------------------------------------------------------- #

def get_df(param_data):
    df = pd.DataFrame(param_data['cap'])
    lista3 = range(0, len(df))
    df.index = lista3
    df['timestamp'] = param_data.index
    param_data = df
    return param_data

def f_estadisticas_mad(param_data, param=False):
    """
      Función que cálcula las estadisticas de los valores de trading(Sharpe, drawdown, drawnup)
      Parameters
      ---------
      param_data: DataFrame: df Data Frame que contiene el profit a utilizar para estas estadisticas
      param: Función de activación para cambiar el nombre de una columna
      Returns
      ---------
      mad: DataFrame: df que contiene las estadisticas
      fecha_max_dd: list: lista donde se encuentra la fecha del valor máximo del Drawdown y su indice
      valor_max_dd: lista: lista donde se encuentra el valor máximo en el DataFrame del Drawdown y su indice
      fecha_min_dd: lista: lista donde se encuentra la fecha del valor mínimo del Drawdown y su indice
      valor_min_dd: lista: lista donde se encuentra el valor mínimo en el DataFrame del Drawdown y su indice
      fecha_max_du: lista: lista donde se encuentra la fecha del valor máximo del Drawup y su indice
      valor_max_du: lista: lista donde se encuentra el valor máximo en el DataFrame del Drawup y su indice
      fecha_min_du: lista: lista donde se encuentra la fecha del valor mínimo del Drawup y su indice
      valor_min_du: lista: lista donde se encuentra el valor mínimo en el DataFrame del Drawup y su indice
      Debuggin
      ---------
      f_estadisticas_mad(data_nueva2)
      """
    # Calculo del Sharpe
    rf = .05 / 300
    daily_closes = param_data['cap']
    daily_logret = np.log(daily_closes / daily_closes.shift()).dropna()
    media = np.mean(daily_logret)
    desvest = np.std(daily_logret)
    if desvest == 0:
        sharpe = 0
    else:
        sharpe = (media - rf) / desvest

    # Drawdown
    # Primera parte: encontrar minimo y analizar su izquierda
    minimo_1_d, indice_minimo_1_d, fecha_min_1_d = get_minimo(param_data, 0)
    if indice_minimo_1_d == 0:
        drawup_1_d = 0
        informacion_drawdown_1 = "[" + " " + str(fecha_min_1_d) + " " + str(fecha_min_1_d) + " " + str(drawup_1_d) + "]"
    else:
        maximo_1_d, indice_maximo_1_d, fecha_max_1_d = get_maximo(param_data.loc[:indice_minimo_1_d], 0)
        drawdown_1 = maximo_1_d - minimo_1_d
        informacion_drawdown_1 = "[" + " " + str(fecha_max_1_d) + " " + str(fecha_min_1_d) + " " + str(drawdown_1) + "]"
    # Segunda parte: encontrar maximo y analizar su derecha
    maximo_2_d, indice_maximo_2_d, fecha_max_2_d = get_maximo(param_data, 0)
    if indice_maximo_2_d == len(param_data):
        drawdown_2 = 0
        informacion_drawup_2 = "[" + " " + str(fecha_max_2_d) + " " + str(fecha_max_2_d) + " " + str(drawdown_2) + "]"
    else:
        minimo_2_d, indice_minimo_2_d, fecha_min_2_d = get_maximo(param_data.loc[indice_maximo_2_d:], indice_maximo_2_d, True)
        drawdown_2 = maximo_2_d - minimo_2_d
        informacion_drawdown_2 = "[" + " " + str(fecha_max_2_d) + " " + str(fecha_min_2_d) + " " + str(drawdown_2) + "]"
    # Elegir el drawdown más grande
    if drawdown_1 > drawdown_2:
        informacion_drawdown = informacion_drawdown_1
        fecha_max_dd = [indice_maximo_1_d, fecha_max_1_d]
        valor_max_dd = [indice_maximo_1_d, maximo_1_d]
        fecha_min_dd = [indice_minimo_1_d, fecha_min_1_d]
        valor_min_dd = [indice_minimo_1_d, minimo_1_d]
    else:
        informacion_drawdown = informacion_drawdown_2
        fecha_max_dd = [indice_maximo_2_d, fecha_max_2_d]
        valor_max_dd = [indice_maximo_2_d, maximo_2_d]
        fecha_min_dd = [indice_minimo_2_d, fecha_min_2_d]
        valor_min_dd = [indice_minimo_2_d, minimo_2_d]
    # Drawup
    # Primera parte: encontrar maximo y analizar su izquierda
    maximo_1, indice_maximo_1, fecha_max_1 = get_maximo(param_data, 0)
    if indice_maximo_1 == 0:
        drawup_1 = 0
        informacion_drawup_1 = "[" + " " + str(fecha_max_1) + " " + str(fecha_max_1) + " " + str(drawup_1) + "]"
    else:
        minimo_1, indice_minimo_1, fecha_min_1 = get_minimo(param_data.loc[:indice_maximo_1], 0)
        drawup_1 = maximo_1 - minimo_1
        informacion_drawup_1 = "[" + " " + str(fecha_min_1) + " " + str(fecha_max_1) + " " + str(drawup_1) + "]"
    # Segunda parte: encontrar minimo y analizar su derecha
    minimo_2, indice_minimo_2, fecha_min_2 = get_minimo(param_data, 0)
    if indice_minimo_2 == len(param_data):
        drawup_2 = 0
        informacion_drawup_2 = "[" + " " + str(fecha_min_2) + " " + str(fecha_min_2) + " " + str(drawup_2) + "]"
    else:
        maximo_2, indice_maximo_2, fecha_max_2 = get_maximo(param_data.loc[indice_minimo_2:], indice_minimo_2, True)
        drawup_2 = maximo_2 - minimo_2
        informacion_drawup_2 = "[" + " " + str(fecha_min_2) + " " + str(fecha_max_2) + " " + str(drawup_2) + "]"
    # Elegir el drawup más grande
    if drawup_1 > drawup_2:
        informacion_drawup = informacion_drawup_1
        fecha_max_du = [indice_maximo_1, fecha_max_1]
        valor_max_du = [indice_maximo_1, maximo_1]
        fecha_min_du = [indice_minimo_1, fecha_min_1]
        valor_min_du = [indice_minimo_1, minimo_1]
    else:
        informacion_drawup = informacion_drawup_2
        fecha_max_du = [indice_maximo_2, fecha_max_2]
        valor_max_du = [indice_maximo_2, maximo_2]
        fecha_min_du = [indice_minimo_2, fecha_min_2]
        valor_min_du = [indice_minimo_2, minimo_2]

    # Crear DataFrame con los datos solicitados
    metricas = ['sharpe', 'drawdown_capi', 'drawup_capi']
    valor = [sharpe, informacion_drawdown, informacion_drawup]
    descripcion = ['Sharpe Ratio', 'DrawDown de Capital', 'DrawUp de Capital']
    mad = pd.DataFrame(columns=['metrica', 'valor', 'descripcion'])
    mad['metrica'] = metricas
    mad['valor'] = valor
    mad['descripcion'] = descripcion

    return mad, [fecha_max_dd, valor_max_dd, fecha_min_dd, valor_min_dd, fecha_max_du, valor_max_du, fecha_min_du,
                 valor_min_du]



