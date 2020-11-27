
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
import plotly.express as px
from data import *
import plotly.graph_objects as go

# -- ---------------------------------------------------------------------------------------------------------------- #
'''--------------------------------------------------------------
Datos históricos de divisa
'''

datos_divisa = read_pkl('USD_MXN')  # 4HRS --> USD/MXN - Mexican Peso
# print(datos_divisa)
# features_divisa = ft.f_features(datos_divisa, 3)
# -- ---------------------------------------------------------------------------------------------------------------- #
# Grafica del periodograma


def get_periodogram(data):
    f, pxx_den = signal.periodogram(data['Open'], 1)
    fig = px.line(x=1/f, y=pxx_den)
    fig.update_layout(title="Periodicidad", xaxis_title="Periodo", yaxis_title="PSD")
    return fig.show()

# -- ---------------------------------------------------------------------------------------------------------------- #
# Grafica para encontrar atípicos


def get_atipicos(data):
    data = data['Open']
    train_unique, counts = np.unique(data, return_counts=True)

    colors = ['Blue: Datos en rango'] * len(train_unique)
    colors[-1] = 'Red: Valores atípicos'

    fig = px.scatter(x=train_unique, y=np.ones(len(train_unique)), color=colors)
    fig.update_layout(shapes=[
        {'type': 'line', 'yref': 'paper', 'y0': train_unique, 'y1': train_unique, 'xref': 'x', 'x0': 17, 'x1': 22}
    ])
    fig.update_layout(title="Detección de valores atípicos", xaxis_title="Prices")
    return fig.show()

