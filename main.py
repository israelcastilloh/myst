
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from data import *

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Datos históricos de divisa e índice
'''

datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
datos_indice = read_pkl('US30_USD')  # 4HRS --> US30_USD - US Wall St. 30

# -- ---------------------------------------------------------------------------------------------------------------- #
