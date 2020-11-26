
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
import numpy as np

# -- ---------------------------------------------------------------------------------------------------------------- #
# Grafica del periodograma


def get_periodogram(data):
    f, pxx_den = signal.periodogram(data['Open'], 1)
    plt.plot(1/f, pxx_den)
    plt.xlabel('periodo')
    plt.ylabel('PSD')
    plt.show()
    return f, pxx_den

# -- ---------------------------------------------------------------------------------------------------------------- #
# Grafica para encontrar atípicos


def get_atipicos(data):
    data = data['Open']
    train_unique, counts = np.unique(data, return_counts=True)

    sizes = counts * 100
    colors = ['blue'] * len(train_unique)
    colors[-1] = 'red'

    plt.axhline(1, color='k', linestyle='--')
    plt.scatter(train_unique, np.ones(len(train_unique)), s=sizes, color=colors)
    plt.yticks([])
    return plt.show()

