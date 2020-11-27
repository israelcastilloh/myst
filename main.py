
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

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Datos históricos de divisa
'''

datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
# print(datos_divisa)
# features_divisa = ft.f_features(datos_divisa, 3)
# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Partición de datos en train y test
'''

test = datos_divisa[:'01-01-2019']
train = datos_divisa['01-01-2019':]

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Aspectos estadisticos de la serie de tiempo 
'''

# ciclos = ft.check_seasonal(train)  # Matriz que muestra los ciclos que se repiten
# Obtener la gráfica de atípicos
vs.get_atipicos(train)
# Calcular los valores necesarios para el análisis estadístico
estacionaridad, autocorrelacion, normalidad, seasonal = ft.get_statistics(train)
# Obtener DataFrame con resultados
estadisticos = ft.get_dfestadisticos(estacionaridad, autocorrelacion, normalidad, seasonal)

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Crear variables artificiales - indicadores financieros y estadísticos
'''

features_divisa = ft.add_all_features(datos_divisa)

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Crear variables artificiales - transformaciones matemáticas y change point
'''

#datos_divisa = ft.math_transformations(datos_divisa)

cor_mat = features_divisa.iloc[:, :].corr()

lm_model = ft.mult_regression(p_x=features_divisa.iloc[:, 1:], p_y=features_divisa.iloc[:, 0])

# RSS of the model with all the variables
print('Modelo Lineal 1: rss: ', lm_model['rss'])

# R^2 of the model
print('Modelo Lineal 1: score: ', lm_model['score'])


# Generacion de un feature formado con variable simbolica
symbolic = ft.symbolic_features(p_x=features_divisa.iloc[:, 1:], p_y=features_divisa.iloc[:, 0])

#symbolic['model']._best_programs[3].__str__()

# -- Transformer -- #
nuevos_features = pd.DataFrame(symbolic['fit'], index=features_divisa.index)
nuevos_features_c = pd.concat([features_divisa, nuevos_features], axis=1)

# -- ---------------------------------------------------------------------------------------- Models fit -- #
cor_mat = nuevos_features_c.iloc[:, :].corr()
# Multple linear regression model
lm_model_s = ft.mult_regression(p_x=nuevos_features_c.iloc[:, 1:], p_y=nuevos_features_c.iloc[:, 0])

# RSS of the model with all the variables
print('Modelo Lineal 2: rss: ', lm_model_s['rss'])
# R^2 of the model
print('Modelo Lineal 2: score: ', lm_model_s['score'])


