
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

# -- ASPECTOS ESTADISTICOS ------------------------------------------------------------------------------------------ #
# -- ---------------------------------------------------------------------------------------------------------------- #


def check_stationarity(param_data):
    """
    Funcion que verifica si la serie es estacionaria o no, en caso de que no lo sea se realizan diferenciaciones.
    Solo se permiten 3 diferenciaciones para no sobrediferenciar la serie.
    Parameters
    ---------
    param_data:
    Returns
    ---------
    stationary_series:
    lags:
    Debuggin
    ---------
    stationary_series, lags = check_stationarity()
    """
    # Utilizar prueba dicky fuller para saber si es estacionaria
    test_results = sm.tsa.stattools.adfuller(param_data)
    # Cuando se cumple la siguiente condicion se considera que la serie es estacionaria
    if test_results[0] < 0 and test_results[1] <= 0.05:
        stationary_series = param_data
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
                stationary_series = new_data
                lags = i
                break
            else:
                param_data = new_data
    return stationary_series, lags

# -- ---------------------------------------------------------------------------------------------------------------- #
