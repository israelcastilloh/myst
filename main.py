
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: PROYECTO FINAL DE LA CLASE DE TRADING                                                       -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from data import *

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Datos históricos de divisa
'''
#save_pkl('USD_MXN')
datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
print(datos_divisa)
# -- ---------------------------------------------------------------------------------------------------------------- #
