
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
import seaborn as sns
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


def pronostico(pronos):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pronos.index, y=pronos["predicted"], mode='lines', name='pronóstico'))
    fig.add_trace(go.Scatter(x=pronos.index, y=pronos["real"], mode='lines', name='real'))
    fig.update_layout(title="Pronóstico divisa USD-MXN", xaxis_title="Tiempo (4H)", yaxis_title="Costo Dólar (Pesos)")
    fig.show()


def general(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data["High"], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=data.index, y=data["Low"], mode='lines', name='Low'))
    fig.update_layout(title="Divisa USD-MXN", xaxis_title="Tiempo (D)", yaxis_title="Costo Dólar (Pesos)")
    fig.show()

# -- ---------------------------------------------------------------------------------------------------------------- #
# Graficas Mad


def get_chart_drawdown_drawup(param: False, lista, datos):
    """
    Funcion que te retorna la gráfica de drawdown y drawup de los movimientos diarios de la cuenta de trading.
    Parameters
     ---------
    param: Funcion de activacion: En caso de que de True se activará que retorne la grafica.
    A pesar de que no se le envian parametros a la funcion es importante saber que se utilizan parametros importados
    de functions especificamente de la funcion que calcula el drawdown y drawup, ya que es de ahi donde se obtienen
    los datos para poder graficar correctamente.
    Returns
    ---------
    Grafica con la evolucion del capital diario así como su drawup y su drawdown
    Debugging
    ---------
    get_chart_drawdown_drawup(True)
    """
    if param:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos['timestamp'], y=datos['cap'], mode='lines',
                                 name='Profit diario', line=dict(color='black')))
        # Drawdown

        fig.add_trace(go.Scatter(x=[lista[0][1], lista[2][1]], y=[lista[1][1], lista[3][1]],
                                 mode='lines', name='Drawdown', line=dict(color="crimson", width=4, dash="dashdot",)))
        # Drawup
        fig.add_trace(go.Scatter(x=[lista[4][1], lista[6][1]], y=[lista[5][1], lista[7][1]],
                                 mode='lines', name='Drawdup', line=dict(color="LightSeaGreen", width=4, dash="dashdot",)))
        fig.update_layout(title="Drawdown y Drawup", xaxis_title="Fechas", yaxis_title="Profit acumulado")
        return fig.show()


def heatmap_corr(df):
    '''
    :param df: dataframe contains all the transformations and predictions
    :return: graph
    '''
    plt.figure(figsize=(16, 8))

    #set the limits
    heatmap = sns.heatmap(df.corr(), cmap="YlGnBu", vmin=-1, vmax=1,annot=False)

    ht = heatmap.set_title('Mapa de calor de correlaciones', fontdict={'fontsize': 12}, pad=12);

    return ht