
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: PROYECTO FINAL DE LA CLASE DE TRADING                                                   -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import data_oanda
import pandas as pd

'''--------------------------------------------------------------
Funcion para descargar los datos históricos de OANDA.
- Checar data_oanda.py
'''


def download(instrument, f_inicio = '2016-01-01', f_fin = '2020-01-02', freq = 'D'):
    # Download prices from Oanda into df_pe
    instrumento = instrument

    f_inicio = pd.to_datetime(f_inicio+' 17:00:00').tz_localize('GMT')
    f_fin = pd.to_datetime(f_fin+' 17:00:00').tz_localize('GMT')

    token = data_oanda.access_token

    df_pe = data_oanda.getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran = freq,
                           p3_inst = instrumento, p4_oatk = token, p5_ginc = 4900)
    df_pe = df_pe.set_index('TimeStamp') #set index to date

    df_pe = pd.DataFrame(df_pe.values.astype(float),
                         columns = df_pe.columns,
                         index = df_pe.index)
    return df_pe


'''--------------------------------------------------------------
Funciones para guardar y leer
los históricos en formato pkl

+ save_pkl('USD_MXN')

'''


def save_pkl(ticker):
    data_df = download(str(ticker))
    data_df.to_pickle('./files/' + str(ticker) + '.pkl')
    return


def read_pkl(name):
    return pd.read_pickle('./files/' + str(name) + '.pkl')

'''--------------------------------------------------------------
Funciones de manejo de datos.
'''


def getReturns(data):
    """
    :param data: pandas dataframe containing OHLC prices
    """
    return np.log(data / data.shift(1)).iloc[1:]
