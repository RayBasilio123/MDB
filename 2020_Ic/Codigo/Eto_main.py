import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from Modelos.Eto_experiments import get_x2, get_x30,resource_selection,steps_ahead
from Tratamento.variaveis import latitude_2,  altitude_2, sigma, G, Gsc
from Tratamento.Eto_generator import gera_serie
from Modelos.Eto_models import arvores
import os.path
import seaborn as sns 
import matplotlib as plt
import matplotlib.pyplot as plt

# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
path= 'C:/Users/Ray/Documents/'

data_Ray = pd.read_csv(path+'IC/2020_Ic/Dados/Dados_PP_Eto.csv') 
eto_Ray = gera_serie(data_Ray, latitude_2, altitude_2, Gsc, sigma, G)
data_Ray["Eto"] = eto_Ray
atributeR = ["Eto", "temp_media", "temp_max","radiacao"]
resultado = get_x30(data_Ray, atributeR, ['Eto'])
if(os.path.exists(path+'IC/2020_Ic/Dados/Tabela30R_Lags.csv')):
  print("CSV Ray já existente !")
else:
  print("Exportando dados Ray ...")
  resultado[0].to_csv(path+'IC/2020_Ic/Dados/Tabela30R_Lags.csv')
  print("Concluido !")


data_Patricia_Eto = pd.read_csv(path+'IC/2020_Ic/Dados/ETo_setelagoas.csv') 

data_Patricia= pd.read_csv(path+'IC/2020_Ic/Dados/variaveis_setelagoas.csv')
data_Patricia["Eto"] = data_Patricia_Eto["Eto"]
if(os.path.exists(path+'IC/2020_Ic/Dados/data_PEto.csv')):
  print("JÁ EXISTE")
else:
  data_Patricia.to_csv('IC/2020_Ic/Dados/data_PEto.csv')

atributeP= [ "Tmax","Tmean","I","UR","V","Tmin","J"]
resultadoP=get_x30(data_Patricia,[atributeP],['Eto'])
if(os.path.exists('./Dados/Tabela30P_Lags.csv')):
  print("CSV patricia já existente !")
else:
  print("Exportando dados Patricia ...")
  resultadoP[0].to_csv(path+'IC/2020_Ic/Dados/Tabela30P_Lags.csv')
  print("Concluido !")

# resultados = arvore(data_Patricia)

# sns.boxplot (y = resultados['erro_rmse'] ); 
# plt.xlabel("Horizontes de previsão um dia", fontsize=14)  
# plt.ylabel("RMSE", fontsize=14)

# plt.show ()
arvore_parametros=[
                    [[['J']],[[0]],['Eto'],[1,2,3]],
                    [[['Tmax', 'I']],[[1],[1]],['Eto'],[1,2,3]],
                    [[['J', 'I']],[[0],[1]],['Eto'],[1,2,3]],
                    [[[ "Tmax","J","I"]],[[1],[0],[1]],['Eto'],[1,2,3]],
                    [[[ "Tmax","J","Tmean"]],[[1],[0],[1]],['Eto'],[1,2,3]],
                    [[["J"]],[[0]],['Eto'],[3]],
                    [[[ "J","Tmax"]],[[0],[3]],['Eto'],[3]],
                    [[["J","Tmax","I"]],[[0],[3],[3]],['Eto'],[3]],
                    [[["Tmax"]],[[3]],['Eto'],[3]],
                    [[["J"]],[[0]],['Eto'],[7]],
                    [[[ "J","Tmax"]],[[0],[7]],['Eto'],[7]],
                    [[["J","Tmax","I"]],[[0],[7],[7]],['Eto'],[7]],
                    [[["Tmax"]],[[7]],['Eto'],[7]],
                    [[["J"]],[[0]],['Eto'],[10]],
                    [[[ "J","Tmax"]],[[0],[10]],['Eto'],[10]],
                    [[[ "J","Tmax","I"]],[[0],[10],[10]],['Eto'],[10]],
                    [[["Tmax"]],[[10]],['Eto'],[10]]
                   ]
                   
# melhores_recursos = selecao_recursos(resultadoP[0],data_Patricia)
# arvore_parametros = dias_a_frente(dias,melhores_recursos)

lista2=['Tmax', 'Tmin', 'I', 'Tmean', 'UR', 'V', 'J']

melhores_recursos = resource_selection(data_Patricia,lista2,[],["Eto"],[]) 

arvore_parametros2=steps_ahead(3,melhores_recursos[0],10)
arvore_parametros3=steps_ahead(7,melhores_recursos[1],5)



arvores(data_Patricia,[arvore_parametros2,arvore_parametros3])

print(arvore_parametros2)