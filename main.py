
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: PROYECTO FINAL DE LA CLASE DE TRADING                                                       -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import functions as ft
import visualizations as vs
from data import *

# -- --------------------------------------------------------------------------------------------------------paso 1-- #
'''--------------------------------------------------------------
Datos históricos de divisa
'''

datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
# print(datos_divisa)
# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Partición de datos en train y test
'''

train = datos_divisa[:'01-01-2019']
test = datos_divisa['01-01-2019':]

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Aspectos estadisticos de la serie de tiempo
'''

ciclos = ft.check_seasonal(train)  # Matriz que muestra los ciclos que se repiten
# Obtener la gráfica de atípicos
# Calcular los valores necesarios para el análisis estadístico
estacionaridad, autocorrelacion, normalidad, seasonal, atipicos = ft.get_statistics(train)
# Obtener DataFrame con resultados
estadisticos = ft.get_dfestadisticos(estacionaridad, autocorrelacion, normalidad, seasonal, atipicos)

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Crear variables artificiales - indicadores financieros y estadísticos
'''
features_divisa = ft.f_features(datos_divisa, 3)

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Crear modelo inicial
'''
lm_model = ft.mult_reg(p_x=features_divisa.iloc[:, 1:][:'01-01-2019'], p_y=features_divisa.iloc[:, 0][:'01-01-2019'])

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Crear modelo con features simbolicos
'''
# Generacion de un features simbolicas, agregadas al modelo
symbolic = ft.symbolic_features(p_x=features_divisa.iloc[:, 1:], p_y=features_divisa.iloc[:, 0])
nuevos_features = pd.DataFrame(symbolic['fit'], index=features_divisa.index)

# modelo
lm_model_s = ft.mult_reg(p_x=nuevos_features[:'01-01-2019'],
                         p_y=features_divisa.iloc[:, 0][:'01-01-2019'])

prediccion = ft.recursivo(nuevos_features, features_divisa, lm_model_s["ridge"]["model"]) #reales y pronostico

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Backtest
'''
backtest = ft.backtest(prediccion, datos_divisa)

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Metricas de atribucion al desempeño
'''
residuos = ft.get_residuos(backtest)
hetero = ft.check_hetero(residuos)
df = ft.get_df(backtest)
mad, lista = ft.f_estadisticas_mad(df, True)

