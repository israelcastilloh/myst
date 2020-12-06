## Description
### Modelo de Predicción de Precios Mediante Regresión Simbólica - USD/MXN.

Modelo de regresión simbólica para pronósticar el precio del instrumento USD/MXN (Divisa). Utilizando variables endógenas y algunas transformaciones estadísticas.

## Install dependencies

Install all the dependencies stated in the requirements.txt file, just run the following command in terminal:

        pip3 install -r requirements.txt

Or you can manually install one by one using the name and version in the file.

## Funcionalities

    datos_divisa = read_pkl('USD_MXN')
Descarga de datos USD_MXN

    estadisticos = ft.get_dfestadisticos(estacionaridad, autocorrelacion, normalidad, seasonal, atipicos)
Dataframe con información estadística general de los datos de entrenamiento

    lm_model = ft.mult_reg(p_x,p_y)}
Creación de modelo con las variables que se tengan, modelo de regresión lineal comparado a lineal con regularizaciones lasso, ridge y elastcnet

    backtest = ft.backtest(prediccion, datos_divisa)
Modelo en datos de prueba en una simulado ejercicio de trading

## Author
Araceli Castillo Fuhr. ING. FINANCIERO

Israel Castillo Herrera. ING. FINANCIERO

Francisco Enriquez Muñoz. ING. FINANCIERO

Diana Laura Ramírez Hinojosa. ING. FINANCIERO

## License
**GNU General Public License v3.0**

*Permissions of this strong copyleft license are conditioned on making available
complete source code of licensed works and modifications, which include larger
works using a licensed work, under the same license. Copyright and license notices
must be preserved. Contributors provide an express grant of patent rights.*

## Contact
*For more information in reggards of this repo, please contact name@email.com*
