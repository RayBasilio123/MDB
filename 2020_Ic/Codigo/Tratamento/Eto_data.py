#imports
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from datetime import datetime
import itertools
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
import math

# from variaveis import 
from variaveis import altitude_1,longitude_1,latitude_1
# constantes
latitude = latitude_1
longitude = longitude_1
altitude = altitude_1


# configura o tamanho do grafico
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

print("Começando a rodar")

#Recebe o arquivo csv
url= 'https://raw.githubusercontent.com/RayBasilio123/R5/master/inmet_pampulha_2018_2019.CSV'
df = pd.read_csv(url, sep=";", encoding = "ISO-8859-1").fillna(0)

#Refaz o formato da hora
df['Hora'] = df['Hora'].str.replace(r'\D', '')
df['Hora'] = [x[:-2] for x in df['Hora']]

#Acrescenta um 0 a esquerda em VENTO_VEL e substitui ',' por '.'
#df['VENTO_VEL'] = df['VENTO_VEL'].str.zfill(3)
df['VENTO_VEL'] = df['VENTO_VEL'].str.replace(',','.').fillna(0)

#Substitui ',' por '.' em RADIACAO
df['RADIACAO'] = df['RADIACAO'].str.replace(',','.').fillna(0)

#Substitui ',' por '.' em "PRESSAO_ATMOSFERICA" ao nivel da estação
df['PRESSAO_ATMOSFERICA'] = df['PRESSAO_ATMOSFERICA'].str.replace(',','.').fillna(0)

# #Substitui os valores negativos de radiacao
df['RADIACAO'] = [float(x) for x in df['RADIACAO']]
df = df.assign(RADIACAO = df.RADIACAO.where(df.RADIACAO.ge(0))).fillna(0)

#Substitui ',' por '.' em TEMPERATURA_MAX
df['TEMPERATURA_MAX'] = df['TEMPERATURA_MAX'].str.replace(',','.').fillna(0)

#Substitui ',' por '.' em TEMPERATURA_MIN
df['TEMPERATURA_MIN'] = df['TEMPERATURA_MIN'].str.replace(',','.').fillna(0)

#Agrupa data+hora 
source_col_loc = df.columns.get_loc('Data') 
df['datetime'] = df.iloc[:,source_col_loc:source_col_loc+2].apply(
    lambda x: " ".join(x.astype(str)), axis=1)

#Tranforma data+hora em Datetime e cria um novo dataframe
dataFormatada = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H')
d = {'date':dataFormatada, 'ventovel':df['VENTO_VEL'],'pressaoATM':df['PRESSAO_ATMOSFERICA'],'tempMax':df['TEMPERATURA_MAX'],'tempMin':df['TEMPERATURA_MIN'], 'umidade_Relativa':df['UMIDADE_RELATIVA_DO_AR'],'umi_Rel_Max':df['UMIDADE_REL_MAX'], 'umi_Rel_Min':df['UMIDADE_REL_MIN'],'Rad':df['RADIACAO']}
dataFrame = pd.DataFrame(data=d)
frameList = list(dataFrame.date) #list of all dates in 'dataFrame'

#Completa os dados negativos(Ex:-9999) com a media do dado anterior e posterior
for column in (dataFrame.columns):
   if column != 'date' :
    for i in range(len(dataFrame)):
      if float(dataFrame[column][i]) < 0:
        dataFrame[column][i] = (dataFrame[column][i+24])

#print(dataFrame)
#cria todas as datas existentes de serieStart ate serieEnd
serieStart = '2018-01-01 00:00:00' 
serieEnd = '2019-12-31 23:00:00'
date = pd.date_range(start=serieStart, end=serieEnd, freq='1H')
dt = {'date': date}
frameDate = date #list of dates all dates in 'date'


#including the missing dates + nan values
print("Entrando em um loop infinito")
new_dates = []
new_values_ventovel = []
new_values_rad = []
new_values_temp_max=[]	
new_values_temp_min=[]
new_values_umidade_max=[]
new_values_umidade_min=[]
new_values_umidade_Relativa=[]
new_values_pressaoATM=[]
print("Percorrendo o loop")
for i in frameDate:
  if i in frameList:
    new_dates.append(i)
    new_values_ventovel.append(float(dataFrame[dataFrame['date']==i]['ventovel']))
    new_values_rad.append(float(dataFrame[dataFrame['date']==i]['Rad']))
    new_values_temp_max.append(float(dataFrame[dataFrame['date']==i]['tempMax']))
    new_values_temp_min.append(float(dataFrame[dataFrame['date']==i]['tempMin']))
    new_values_umidade_max.append(float(dataFrame[dataFrame['date']==i]['umi_Rel_Max']))
    new_values_umidade_min.append(float(dataFrame[dataFrame['date']==i]['umi_Rel_Min']))
    new_values_umidade_Relativa.append(float(dataFrame[dataFrame['date']==i]['umidade_Relativa']))
    new_values_pressaoATM.append(float(dataFrame[dataFrame['date']==i]['pressaoATM']))
    print("Rodando ...")
  else:
    new_dates.append(i)
    new_values_ventovel.append(np.nan)
    new_values_rad.append(np.nan)
    new_values_temp_max.append(np.nan)
    new_values_temp_min.append(np.nan)
    new_values_umidade_max.append(np.nan)
    new_values_umidade_min.append(np.nan)
    new_values_umidade_Relativa.append(np.nan)
    new_values_pressaoATM.append(np.nan)

print("Saimos")

#transforming data in series end interpolate NaN values
index = pd.DatetimeIndex(new_dates)
series_ventovel = pd.Series(new_values_ventovel, index=index)
for i in range(series_ventovel.shape[0]-1): 
    if np.isnan(series_ventovel[i]):
      series_ventovel[i] = 2
print("Preenchendo series...")
series_radiacao = pd.Series(new_values_rad, index=index).interpolate()
series_temp_max = pd.Series(new_values_temp_max, index=index).interpolate()
series_temp_min = pd.Series(new_values_temp_min, index=index).interpolate()
series_umidade_max = pd.Series(new_values_umidade_max, index=index).interpolate()
series_umidade_min = pd.Series(new_values_umidade_min, index=index).interpolate()
series_umidade_Relativa = pd.Series(new_values_umidade_Relativa, index=index).interpolate()
series_pressaoATM = pd.Series(new_values_pressaoATM, index=index).interpolate()

#Cria novo dataFrame ajustado
print("Ajustando data...")
dataAjustada= pd.DataFrame(index=index)
dataAjustada.index.name = 'Data'
dataAjustada['vento'] = series_ventovel.values
dataAjustada['radiacao'] = series_radiacao.values
dataAjustada['temp_max'] = series_temp_max.values
dataAjustada['temp_min'] = series_temp_min.values
dataAjustada['umi_max'] = series_umidade_max.values
dataAjustada['umi_min'] = series_umidade_min.values
dataAjustada['umi_rel'] = series_umidade_Relativa.values
dataAjustada['press_atm'] = series_pressaoATM.values
dataAjustada.head(24)

dataAjustada.reset_index(level=0,inplace=True)
temp_max_dia=[]	
temp_min_dia=[]
umidade_min_dia =[]
umidade_max_dia =[]
grouped = dataAjustada.groupby(pd.Grouper(key = 'Data', freq = 'D'))
umidade_min_dia=grouped['umi_min'].min()
umidade_max_dia=grouped['umi_max'].max()
temp_min_dia=grouped['temp_min'].min()
temp_max_dia=grouped['temp_max'].max()

dataAjustada.set_index('Data', inplace=True)
print("Criando dataFrameDay ...")
#faz uma media dos dados diarios e os transforma em dias
dataFrameDay = pd.DataFrame()
dataFrameDay = dataAjustada.resample('D').mean()
temp_media = []
count = 0
for i in range (len(dataFrameDay)):
  temp_media.append((dataFrameDay['temp_max'][i] + dataFrameDay['temp_min'][i])/2)

dataFrameDay['temp_media'] = temp_media
dataFrameDay['temp_max'] = temp_max_dia.values
dataFrameDay['temp_min'] = temp_min_dia.values
dataFrameDay['umi_max'] = umidade_max_dia.values
dataFrameDay['umi_min'] = umidade_min_dia.values
print("Preenchendo o dataFrameDay ...")
#Calcula U2
#Transforma a velocidade do vendo para uma velocidade a 2 m de altura e caso não tenha o valor da velocidade considera 
#2 m/s 

for i in range(len(dataFrameDay['vento'])):
 if float(dataFrameDay['vento'][i]) >= 0:
    dataFrameDay['vento'][i] = float(dataFrameDay['vento'][i])* (4.87 / math.log(67.8 * altitude - 5.42))  
    if dataFrameDay['vento'][i]<0.5:
      dataFrameDay['vento'][i]=0.5

 for i in range(len(dataFrameDay)):
        dataFrameDay["radiacao"][i] = (dataFrameDay["radiacao"][i])/(1000000/86400)

#exporta para csv
print("Exportando dataFrameDay ...")

dataFrameDay.to_csv('./Dados/Dados_PP_Eto.csv')

print("Concluido !")