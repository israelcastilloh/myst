
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
from data import *

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Datos históricos de divisa
'''

#save_pkl('USD_MXN')
datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
#print(datos_divisa)

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

estadisticos = ft.df_estadisticos
# -- ---------------------------------------------------------------------------------------------------------------- #
#%%
'''crear variables artificiales - indicadores financieros y estadísticos'''
datos_divisa = ft.add_all_features(datos_divisa)

#%%
'''crear variables artificiales - transformaciones matemáticas y change point'''
datos_divisa = ft.math_transformations(datos_divisa)

