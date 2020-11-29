
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

# ciclos = ft.check_seasonal(train)  # Matriz que muestra los ciclos que se repiten
# Obtener la gráfica de atípicos
#vs.get_atipicos(train)
# Calcular los valores necesarios para el análisis estadístico
estacionaridad, autocorrelacion, normalidad, seasonal = ft.get_statistics(train)
# Obtener DataFrame con resultados
estadisticos = ft.get_dfestadisticos(estacionaridad, autocorrelacion, normalidad, seasonal)

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
nuevos_features_c = pd.concat([features_divisa, nuevos_features], axis=1)

# modelo
lm_model_s = ft.mult_reg(p_x=nuevos_features_c.iloc[:, 1:][:'01-01-2019'], p_y=nuevos_features_c.iloc[:, 0][:'01-01-2019'])

recursivo=ft.recursivo(nuevos_features_c, lm_model_s["ridge"]["model"])

# dataframe con nuevos_features_c["predict"] (dato real que quiero pronosticar),recursivo (dato pronosticado con ridge)
# , close de datos divisa recortado, "decisión"