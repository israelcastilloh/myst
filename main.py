
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

# Historicos de la divisa --> USD/MXN
datos_divisa = read_pkl('USD_MXN')
print(datos_divisa)

# Historicos del índice --> US30_USD - US Wall St. 30
datos_indice = read_pkl('US30_USD')
print(datos_indice)
