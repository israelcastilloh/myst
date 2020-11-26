
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

# save_pkl('USD_MXN')
datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
# print(datos_divisa)

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

ciclos = ft.check_seasonal(train)  # Matriz que muestra los ciclos que se repiten
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

datos_divisa = ft.add_all_features(datos_divisa)

'''--------------------------------------------------------------
crear variables artificiales - transformaciones matemáticas y change point
'''

datos_divisa = ft.math_transformations(datos_divisa)

