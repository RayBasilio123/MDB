import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as plt
import matplotlib.pyplot as plt
from math import nan
from Tratamento.variaveis import latitude_2,  altitude_2, sigma, G, Gsc
from Tratamento.Eto_generator import gera_serie
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



def get_x2(df,lista,lags,Eto,lags_eto):
  lags=lags
  lista_aux=[]
  lista_aux2=[]
  max_lag=0;
  data = pd.DataFrame()
  eta_nois = pd.DataFrame()
  eta_nois['Data']=df['Data']
  for coluna in lista:
    data[coluna] = df[coluna]       

  for i in range(len(lags)):
    for j in range(len(lags[i])):
      lista_aux = data.iloc[:,i].tolist()
      for displacement in range((lags[i][j])):
        if max_lag<(lags[i][j]):
          max_lag=(lags[i][j])
        del lista_aux[len(lista_aux)-1]
        lista_aux.insert(0,nan)
      eta_nois[((data.iloc[:,i]).name)+("_t-")+str(lags[i][j])]=(lista_aux)  
  
  for i in range(len(lags_eto)):
    lista_aux2=df[Eto].iloc[:,0].tolist()
    for displacement in range((lags_eto[i])):
      if max_lag<(lags_eto[i]):
        max_lag=(lags_eto[i])
      del lista_aux2[len(lista_aux2)-1]
      lista_aux2.insert(0,nan)
    eta_nois[((df[Eto].iloc[:,0]).name)+("_t-")+str(lags_eto[i])]=lista_aux2        
  return eta_nois,max_lag,lista,lags,Eto,lags_eto;



def get_x30(df,lista, Eto):
  ix = []
  idx =  [i for i in np.arange(1, 31)]
  for i in range(len(lista)):
    ix.append(idx)

  resultado = get_x2(df, lista, ix, Eto, idx)
  return resultado

  


def pacf_acf(df_patricia,atributeP):
  for i in (atributeP):
    plot_acf(df_patricia[i],lags=60,title="Autocorrelation "+i)
    plot_pacf(df_patricia[i],lags=60,title= "Partial Autocorrelation "+i)
  plt.show();


  

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesRegressor

def resource_selection(df,lista,lags,Eto,lags_eto) :  
  ix = []
  idx =  [i for i in np.arange(1,31)]
  if lista==[]:
    lista=df.columns.drop('Data').drop('Eto')
  if lags==[]:
    for i in range(len(lista)):
      ix.append(idx)
    lags=ix
  if lags_eto==[]:
    lags_eto=idx
  if Eto==[]:
    tab=get_x2(df,lista,lags,Eto,[])
  else:
    tab=get_x2(df,lista,lags,Eto,lags_eto)
  
  df30=tab[0].drop("Data",axis=1)
  df30=df30.iloc[tab[1]:,:]
  print(df30)
  array1 = df30.values
  df= df.drop("Data",axis=1)
  df=df.iloc[tab[1]:,:]
  array2 = df["Eto"]
  X =array1[:,0:len(tab[0].columns)]
  Y = array2
  # feature extraction
  test = SelectKBest(score_func=f_regression, k=4)
  fit = test.fit(X, Y)
  # summarize scores
  set_printoptions(precision=3)
 

    
  # print("Selecao_univariada",fit.scores_)
  f=fit.scores_
  f_ord = sorted(f,reverse=True)
  # print("Selecao_univariada_ordenada",f_ord)

  ll=[]
  for y in range(len(f_ord)):
    for i in range(len(f)):
      if (f_ord[y]==f[i]):
        ll.append(i)
        # print("f[i]",f[i])
        # print("f_ord[y]",f_ord[y])
  leg_seq=[]
  for i in range(len(ll)):
    leg_seq.append(df30.columns[ll[i]])    

  print("\n\nColunas_selecionadas [0]- Selecao_univariada",leg_seq)
 
  model = ExtraTreesRegressor(n_estimators=10)
  model.fit(X, Y)
  g=model.feature_importances_
  g_ord=sorted(model.feature_importances_,reverse=True)
  # print("Importancia",g)
  # print("Importancia_ordenada",g_ord)
  jj=[]
  for y in range(len(g_ord)):
    for i in range(len(g)):
      if (g_ord[y]==g[i]):
        jj.append(i)
  leg_seq2=[]
  for i in range(len(ll)):
    leg_seq2.append(df30.columns[jj[i]])    
  print("\n\nColunas_selecionadas [1]- Importância do recurso",leg_seq2)
  return leg_seq,leg_seq2

def steps_ahead(dias,melhores_recursos,q):
  b=[]
  lags=[]
  aux3=[]
  lista=[]
  Eto=[]
  lags_eto=[]
  # quantidade = len(melhores_recursos) if quantidade==0 else quantidade
  # filtra e  remove colunas 
  for i in range(len(melhores_recursos)):
    j=melhores_recursos[i].split("_t-")
    if (int(j[1])>=dias):
      b.append(j)
  print("lista_selecionada",b)

  #localiza as colunas  e faz uma lista se a coluna for Eto cria uma lista só pra ela
  for k in range(len(b)):
    if b[k][0] not in lista and b[k][0]!="Eto":
        lista.append(b[k][0])
    elif b[k][0] not in Eto and b[k][0]=="Eto":
        Eto.append(b[k][0])

  print("lista",lista)
  print("Eto",Eto)
 
  # localiza as colunas e cria uma lista de lags para cada
  for i in range(len(lista)):
    aux=[]
    for k in range(len(b)):
      if (lista[i]==b[k][0])  :
          aux.append(int(b[k][1]))
    lags.append(aux)
    del(aux)

  print("lags",lags)
           
  # print("Eto",Eto)
  
  
  if Eto!=[]:
    for k in range(len(b)):
      if b[k][0]==Eto[0]:
        lags_eto.append(int(b[k][1]))
  print("lags_eto",lags_eto)
 
  aux3.append(lista)
  aux3.append(lags)
  aux3.append(Eto)
  aux3.append(lags_eto)
 

  return aux3
