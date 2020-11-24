
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: PROYECTO FINAL DE LA CLASE DE TRADIN                                                       -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from scipy import signal
import matplotlib.pyplot as plt
from functions import *

# -- ---------------------------------------------------------------------------------------------------------------- #
# Grafica del periodograma


def get_periodogram(p=False):
    if p:
        f, pxx_den = signal.periodogram(train['Open'], 1)
        plt.plot(1/f, pxx_den)
        plt.xlabel('periodo')
        plt.ylabel('PSD')
        plt.show()
        return f, pxx_den
