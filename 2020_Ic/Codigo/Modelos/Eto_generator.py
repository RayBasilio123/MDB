
"""
Gera série temporal de Evapotranspiração de Referência
Referência: FAO 56 (2006)
Dados climáticos usados como entrada:
- Temperaturas (máxima, mínima e média) do ar em °C - Tmax, Tmin e Tmean
- Umidade Relativa Média - RH
- Insolação em Horas - I
- Velocidade do vento em m/s - U2
- Dia do ano - J
Dados da estação meteorológica usados como entrada:
Latitude em radianos
Altitude em metros
Constante Solar é de 0.0820  MJ m−2 min−1 
Constante de Stefan Boltzmann é de 0.000000004903  MJ m−2 dia−1
Fluxo de calor do solo (G) para o período de 1 dia ou 10 dias = 0
"""



import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as plt
import matplotlib.pyplot as plt



#Constante Psicrometrica
def gamma(altitude):
  eve = pow(((293-0.0065*altitude)/293),5.26)
  p =101.3*eve
  gamma=p*0.000663
  return gamma

#Pressão de valor de saturação
def Es(t):
    return 0.6108 * math.exp((17.27 * t) / (t + 237.3))
def Es_medio(tmin, tmax):    
    return (Es(tmin) + Es(tmax)) / 2.0

#Pressão de vapor
def Ea(Temp_max,Temp_min,UR):
  ea=(UR* Es_medio(Temp_max,Temp_min))/100
  return ea

#Declividade da curva de pressão de vapor dágua da atm
def Delta(Temp_max,Temp_min):
  T = (Temp_max+Temp_min)/2
  aux_delta = 4098*pow(0.6108, (17.27 * T)/(T + 237.3) )
  delta = aux_delta/(pow((T+237.3),2))
  return aux_delta

def Rns(rs, albedo=0.23):  
    rns =  (1 - albedo) * rs
    return rns

# Radiação solar que atinge a superficie da terra
def Rs(N, Ra, Temp_max, Temp_min):
    rs = 0.16 * np.sqrt(Temp_max - Temp_min) * Ra
    return rs

def Rso(altitude, ra):
    rso = (0.00002 * altitude + 0.75) * ra
    return rso

#Diferença entre a radiação de onda longa emitida e recebida 
def Rnl(Temp_max, Temp_min, rs, rso, ea, sigma):  
    k_temp_max = Temp_max + 273.16 
    k_temp_min = Temp_min + 273.16     
    rnl = (sigma * ((pow(k_temp_max, 4) + pow(k_temp_min, 4)) / 2))*(0.34 - (0.14 * math.sqrt(ea)))*( 1.35 * (rs / rso) - 0.35)
    return rnl
    
#Saldo de radiação na superficie da cultura
def Rn(Rns,Rnl):
  rn=Rns-Rnl
  return rn

def Et_fao(rn, t, u2, es, ea, delta, gamma, G):
    a1 = (0.408 * (rn - G) * delta) + ((900 / (t + 273)) * u2 * gamma * (es - ea))
    a2 =  a1 / (delta + (gamma * (1 + 0.34 * u2)))
    return a2

def N_insolacao(omega):
   insol = (24.0 / math.pi) * omega
   return insol

def Dr(J):
    dr = 1 + (0.033 * math.cos((2.0 * math.pi / 365.0) * J))
    return dr

def Omega(latitude, declinacao_sol):
    cos_sha = -math.tan(latitude) * math.tan(declinacao_sol)
    omega = math.acos(min(max(float(cos_sha), float(-1.0)), float(1.0)))
    return omega
    
def Declinacao_sol(J):
    decli_sol = float(0.409) * math.sin(((float(2.0) * math.pi / float(365.0)) * J - float(1.39)))
    return decli_sol

def Ra(latitude, declinacao_sol, omega, dr, Gsc):
    tmp1 = (24.0 * 60.0) / math.pi
    tmp2 = omega * math.sin(latitude) * math.sin(declinacao_sol)
    tmp3 = math.cos(latitude) * math.cos(declinacao_sol) * math.sin(omega)
    rad = tmp1 * Gsc * dr * (tmp2 + tmp3)
    return rad

def media_umi(umi_max, umi_min):
  media = (umi_max + umi_min)/2
  return media
  
def Pressao_atm(altitude):
    """
    Pressão Atmosférica (P): Equação 7 (FAO 56)
    :parâmetro altitude: altitude acima do nível do mar [m]
    :return: pressão atmosférica [kPa]
    """
    tmp = (293.0 - (0.0065 * altitude)) / 293.0
    return math.pow(tmp, 5.26) * 101.3

def calcula_dia(dataset):
    """
      Calcula dia do ano e acrescenta na base de dados
      :param dataset: base de dados completa
      :return: base de dados + coluna com o dia do ano
    """
    date = dataset['Data']
    day_of_year = []
    for i in range(date.shape[0]):
      adate = datetime.strptime(date[i],"%Y-%m-%d")
      day_of_year.append(adate.timetuple().tm_yday)
    day = np.asarray(day_of_year)
    dayframe=pd.DataFrame(day,columns=['J'])
    d = [dataset,dayframe]
    dataset = pd.concat(d,axis=1)
    return dataset
    
def gera_serie(dataset, latitude, altitude, Gsc, sigma, G):

    serie_eto = []
    for linha in range(len(dataset)):
        es = Es_medio(dataset.iloc[linha,3],dataset.iloc[linha,4]) #------------> Pressão do vapor de saturação
        media =  media_umi(dataset.iloc[linha, 5], dataset.iloc[linha, 6])
        ea = Ea(dataset.iloc[linha,3],dataset.iloc[linha,4],media) #--------> Pressão do vapor atual
        # print('i {}, tmin {}, tmax {}, media {}, ea {}'.format(linha,dataset.iloc[linha,3],dataset.iloc[linha,4],media,ea))
        
        delta = Delta(dataset.iloc[linha,3], dataset.iloc[linha, 4]) #----------------------> Declividade da curva de pressão do vapor
        pressao_atm = Pressao_atm(altitude) #-----------> Pressão atmosférica
        gamma_ = gamma(altitude) #------------> Constante
        declinacao_sol = Declinacao_sol(dataset.iloc[linha,10]) #----> Declinação solar
        omega = Omega(latitude, declinacao_sol) #-------> Ângulo horário pôr-do-sol
        dr = Dr(dataset.iloc[linha,10]) #----------------------------> Inverso da distância relativa da terra-sol
        ra = Ra(latitude, declinacao_sol, omega, dr, Gsc) #--> Radiação extraterrestre para períodos diários
        N = N_insolacao(omega) #------------------------> Duração máxima de insolação no dia
        rs = Rs(N, ra, dataset.iloc[linha,3], dataset.iloc[linha,4]) #---------------------> Radiação solar
        rso = Rso(altitude, ra) #-----------------------> Radiação solar de céu claro
        rns = Rns(rs, albedo=0.23) #--------------------> Radiação de onda curta líquida
        rnl = Rnl(dataset.iloc[linha,3],dataset.iloc[linha,4], rs, rso, ea, sigma) #---> Radiação de onda longa líquida
        rn = Rn(rns,rnl) #------------------------------> Radiação líquida
        serie_eto.append(Et_fao(rn, dataset.iloc[linha,9], dataset.iloc[linha,1], es, ea, delta, gamma_, G=0)) #---> Evapotranspiração
      #  (rn, t, u2, es, ea, delta, gamma, G)
        
    return serie_eto



