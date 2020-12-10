#building the tree
# Import the necessary modules and libraries
import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score
from Modelos.Eto_experiments import gera_serie
from Modelos.Eto_experiments import get_x2

# df_patricia = pd.read_csv('C:/Users/Ray/Documents/2020_Ic/Dados/data_PEto.csv') 
# df_patricia
def arvore(df_patricia):
    print("Calculando o fit e o predict...")

    list_lags_etoACF=[1,2,3,4,5]
    list_lags_etoPACF=[1,2,3,4,5]
    list_lags_TmaxACF=[1,2,3,4,5]
    list_lags_TmaxPACF=[1,3,7,5,4]
    list_lags_TmenACF=[1,2,3,4,5]
    list_lags_TmenPACF=[1,3,5,7,6]
    list_lags_IACF=[1,2,3,4,5]
    list_lags_IPACF=[1,2,4,5,21]
    atributeP = [ "Tmax","Tmean","I","UR","V","Tmin","J"]
    print("Calculando o get_x2 e acrescentando em tabelinhas")

    # tabelinha1=get_x2(df_patricia,atributeP,[list_lags_TmaxPACF],['Eto'],list_lags_etoACF)
    # tabelinha2=get_x2(df_patricia,['Tmax', 'Tmean'],[list_lags_TmaxPACF,list_lags_TmenPACF],['Eto'],[])
    # tabelinha3=get_x2(df_patricia,[  'Tmax', 'Tmean', 'I'],[list_lags_TmaxPACF,list_lags_TmenPACF,list_lags_IPACF],['Eto'],[])
    # tabelinha4=get_x2(df_patricia,[ 'Tmean', 'I'],[list_lags_TmenPACF,list_lags_IPACF],['Eto'],[])
    # tabelinha5=get_x2(df_patricia,[ 'Tmean', 'I'],[list_lags_TmenPACF,list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha6=get_x2(df_patricia,['Tmean'],[list_lags_TmenPACF],['Eto'],list_lags_etoPACF)
    # tabelinha7=get_x2(df_patricia,['I'],[list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha8=get_x2(df_patricia,['Tmax'],[list_lags_TmaxPACF],['Eto'],list_lags_etoPACF)
    # tabelinha9=get_x2(df_patricia,['Tmax', 'Tmean'],[list_lags_TmaxPACF,list_lags_TmenPACF],['Eto'],list_lags_etoPACF)
    # tabelinha10=get_x2(df_patricia,[  'Tmax', 'Tmean', 'I'],[list_lags_TmaxPACF,list_lags_TmenPACF,list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha11=get_x2(df_patricia,[ 'Tmean', 'I'],[list_lags_TmenPACF,list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha12=get_x2(df_patricia,[ 'Tmax', 'I'],[list_lags_TmaxPACF,list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha13=get_x2(df_patricia,['I'],[list_lags_IPACF],['Eto'],list_lags_etoPACF)
    # tabelinha14=get_x2(df_patricia,['Tmean'],[list_lags_TmenPACF],['Eto'],list_lags_etoPACF)
    # tabelinha15=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[0],[0],[0],[0],[0]],['Eto'],[0])
    
    #lags zero e c/eto e s/eto
    
    tabelinha1=get_x2(df_patricia,[['J']],[[0]],['Eto'],[1,2,3])
    tabelinha2=get_x2(df_patricia,[['Tmax', 'I']],[[1],[1]],['Eto'],[1,2,3])
    tabelinha3=get_x2(df_patricia,[[ "Tmax","J","I"]],[[1],[0],[1]],['Eto'],[1,2,3])
    tabelinha4=get_x2(df_patricia,[["J"]],[[0]],['Eto'],[3])
    tabelinha5=get_x2(df_patricia,[[ "J","Tmax"]],[[0],[3]],['Eto'],[3])
    tabelinha6=get_x2(df_patricia,[["J","Tmax","I"]],[[0],[3],[3]],['Eto'],[3])
    tabelinha7=get_x2(df_patricia,[[ "Tmax"]],[[3]],['Eto'],[3])
    tabelinha8=get_x2(df_patricia,[['Tmax', 'Tmean']],[[0],[0]],['Eto'],[])
    tabelinha9=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[0],[0],[0]],['Eto'],[])
    tabelinha10=get_x2(df_patricia,[[ "Tmax","Tmean","J"]],[[0],[0],[0]],['Eto'],[])
    tabelinha11=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[0],[0],[0],[0]],['Eto'],[0])
    tabelinha12=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[0],[0],[0],[0],[0]],['Eto'],[])
    tabelinha13=get_x2(df_patricia,[['Tmean']],[[0]],['Eto'],[0])
    tabelinha14=get_x2(df_patricia,[[ 'Tmean',"I"]],[[0],[0]],['Eto'],[0])
    tabelinha15=get_x2(df_patricia,[[ "Tmean","I","UR"]],[[0],[0],[0]],['Eto'],[0])
    
    # tabelinha1=get_x2(df_patricia,[[ "Tmean","I","UR","J"]],[[0],[0],[0],[0]],['Eto'],[0])
    # tabelinha2=get_x2(df_patricia,[[ "Tmean","I","V"]],[[0],[0],[0]],['Eto'],[0])
    # tabelinha3=get_x2(df_patricia,[[ "Tmean","I","J","UR","V"]],[[0],[0],[0],[0],[0]],['Eto'],[0])
    # tabelinha4=get_x2(df_patricia,[['Tmean']],[[0]],['Eto'],[])
    # tabelinha5=get_x2(df_patricia,[[ 'Tmean',"I"]],[[0],[0]],['Eto'],[])
    # tabelinha6=get_x2(df_patricia,[[ "Tmean","I","UR"]],[[0],[0],[0]],['Eto'],[])
    # tabelinha7=get_x2(df_patricia,[[ "Tmean","I","UR","J"]],[[0],[0],[0],[0]],['Eto'],[])
    # tabelinha8=get_x2(df_patricia,[[ "Tmean","I","V"]],[[0],[0],[0]],['Eto'],[])
    # tabelinha9=get_x2(df_patricia,[[ "Tmean","I","J","UR","V"]],[[0],[0],[0],[0],[0]],['Eto'],[])
    # tabelinha10=get_x2(df_patricia,[['I']],[[0]],['Eto'],[0])
    # tabelinha11=get_x2(df_patricia,[['I', 'UR']],[[0],[0]],['Eto'],[0])
    # tabelinha12=get_x2(df_patricia,[[ "I",'UR','J']],[[0],[0],[0]],['Eto'],[0])
    # tabelinha13=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[0],[0],[0],[0]],['Eto'],[0])
    # tabelinha14=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[0],[0],[0],[0]],['Eto'],[1])
    # tabelinha15=get_x2(df_patricia,[[ "I","UR","J","Tmin","V"]],[[0],[0],[0],[0],[0]],['Eto'],[0])
    
    # tabelinha3=get_x2(df_patricia,[['I']],[[0]],['Eto'],[])
    # tabelinha4=get_x2(df_patricia,[['I', 'UR']],[[0],[0]],['Eto'],[])
    # tabelinha5=get_x2(df_patricia,[[ "I",'UR','J']],[[0],[0],[0]],['Eto'],[])
    # tabelinha6=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[0],[0],[0],[0]],['Eto'],[])
    # tabelinha7=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[0],[0],[0],[0]],['Eto'],[2])
    # tabelinha8=get_x2(df_patricia,[[ "I","UR","J","Tmin","V"]],[[0],[0],[0],[0],[0]],['Eto'],[])
    
    # tabelinha9=get_x2(df_patricia,[['Tmax']],[[1]],['Eto'],[1])
    # tabelinha10=get_x2(df_patricia,[['Tmax', 'Tmean']],[[1],[1]],['Eto'],[1])
    # tabelinha11=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[1],[1],[1]],['Eto'],[1])
    # tabelinha12=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha13=get_x2(df_patricia,[[ "Tmax","Tmean","I","UR"]],[[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha14=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[1],[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha15=get_x2(df_patricia,[[ "Tmax"]],[[1]],['Eto'],[])
    
    # tabelinha1=get_x2(df_patricia,[['Tmax', 'Tmean']],[[1],[0]],['Eto'],[])
    # tabelinha2=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[1],[1],[1]],['Eto'],[])
    # tabelinha3=get_x2(df_patricia,[[ "Tmax","Tmean","J"]],[[1],[1],[1]],['Eto'],[])
    # tabelinha4=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[1],[1],[1],[1]],['Eto'],[])
    # tabelinha5=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[1],[1],[1],[1],[1]],['Eto'],[])
    # tabelinha6=get_x2(df_patricia,[['Tmean']],[[1]],['Eto'],[1])
    # tabelinha7=get_x2(df_patricia,[[ 'Tmean',"I"]],[[1],[1]],['Eto'],[1])
    # tabelinha8=get_x2(df_patricia,[[ "Tmean","I","UR"]],[[1],[1],[1]],['Eto'],[1])
    # tabelinha9=get_x2(df_patricia,[[ "Tmean","I","UR","J"]],[[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha10=get_x2(df_patricia,[[ "Tmean","I","V"]],[[1],[1],[1]],['Eto'],[1])
    # tabelinha11=get_x2(df_patricia,[[ "Tmean","I","J","UR","V"]],[[1],[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha12=get_x2(df_patricia,[['Tmean']],[[1]],['Eto'],[])
    # tabelinha13=get_x2(df_patricia,[[ 'Tmean',"I"]],[[1],[1]],['Eto'],[])
    # tabelinha14=get_x2(df_patricia,[[ "Tmean","I","UR"]],[[1],[1],[1]],['Eto'],[])
    # tabelinha15=get_x2(df_patricia,[[ "Tmean","I","UR","J"]],[[1],[1],[1],[1]],['Eto'],[])
    
    # tabelinha1=get_x2(df_patricia,[[ "Tmean","I","V"]],[[1],[1],[1]],['Eto'],[])
    # tabelinha2=get_x2(df_patricia,[[ "Tmean","I","J","UR","V"]],[[1],[1],[1],[1],[1]],['Eto'],[])
    # tabelinha3=get_x2(df_patricia,[['I']],[[1]],['Eto'],[1])
    # tabelinha4=get_x2(df_patricia,[['I', 'UR']],[[1],[1]],['Eto'],[1])
    # tabelinha5=get_x2(df_patricia,[[ "I",'UR','J']],[[1],[1],[1]],['Eto'],[1])
    # tabelinha6=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha7=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[1],[1],[1],[1]],['Eto'],[2])
    # tabelinha8=get_x2(df_patricia,[[ "I","UR","J","Tmin","V"]],[[1],[1],[1],[1],[1]],['Eto'],[1])
    # tabelinha9=get_x2(df_patricia,[['I']],[[1]],['Eto'],[])
    # tabelinha10=get_x2(df_patricia,[['I', 'UR']],[[1],[1]],['Eto'],[])
    # tabelinha11=get_x2(df_patricia,[[ "I",'UR','J']],[[1],[1],[1]],['Eto'],[])
    # tabelinha12=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[1],[1],[1],[1]],['Eto'],[])
    # tabelinha13=get_x2(df_patricia,[[ "I","UR","J","Tmin"]],[[1],[1],[1],[1]],['Eto'],[2])
    # tabelinha14=get_x2(df_patricia,[[ "I","UR","J","Tmin","V"]],[[1],[1],[1],[1],[1]],['Eto'],[])
    # tabelinha15=get_x2(df_patricia,[['Tmax']],[[1,2]],['Eto'],[1])

    # tabelinha1=get_x2(df_patricia,[['Tmax', 'Tmean']],[[1,2],[1,2]],['Eto'],[1])
    # tabelinha2=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[1,2],[1,2],[1,2]],['Eto'],[1,2])
    # tabelinha3=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[1,2])
    # tabelinha4=get_x2(df_patricia,[[ "Tmax","Tmean","I","UR"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[1,2])
    # tabelinha5=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[1,2],[1,2],[1,2],[1,2],[1,2]],['Eto'],[1,2])
    
    # tabelinha6=get_x2(df_patricia,[['Tmax']],[[1,2]],['Eto'],[0])
    # tabelinha7=get_x2(df_patricia,[['Tmax', 'Tmean']],[[1,2],[1,2]],['Eto'],[0])
    # tabelinha8=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[1,2],[1,2],[1,2]],['Eto'],[0])
    # tabelinha9=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[0])
    # tabelinha10=get_x2(df_patricia,[[ "Tmax","Tmean","I","UR"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[0])
    # tabelinha11=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[1,2],[1,2],[1,2],[1,2],[1,2]],['Eto'],[0])
    
    # tabelinha12=get_x2(df_patricia,[['Tmax']],[[1,2]],['Eto'],[])
    # tabelinha13=get_x2(df_patricia,[['Tmax', 'Tmean']],[[1,2],[1,2]],['Eto'],[])
    # tabelinha14=get_x2(df_patricia,[[ "Tmax","Tmean","I"]],[[1,2],[1,2],[1,2]],['Eto'],[])
    # tabelinha15=get_x2(df_patricia,[[ "Tmax","Tmean","I","J"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[])
    
    # tabelinha1=get_x2(df_patricia,[[ "Tmax","Tmean","I","UR"]],[[1,2],[1,2],[1,2],[1,2]],['Eto'],[])
    # tabelinha2=get_x2(df_patricia,[[ "Tmax","Tmean","I","J","UR"]],[[1,2],[1,2],[1,2],[1,2],[1,2]],['Eto'],[])
    
    print("Listando tabelas ...")
    lista_tabelas =[tabelinha1,
                    tabelinha2,
                    tabelinha3,
                    tabelinha4,
                    tabelinha5,
                    tabelinha6,
                    tabelinha7,
                    tabelinha8,
                    tabelinha9,
                    tabelinha10,
                    tabelinha11,
                    tabelinha12,
                    tabelinha14,
                    tabelinha15]
    print("Tratando dataFrame...")

    x1 = tabelinha1[0].drop("Data", axis=1)
    y1 = df_patricia["Eto"]
    x2 = tabelinha2[0].drop("Data", axis=1)
    y2 = df_patricia["Eto"]
    x3 = tabelinha3[0].drop("Data", axis=1)
    y3 = df_patricia["Eto"]
    x4 = tabelinha4[0].drop("Data", axis=1)
    y4 = df_patricia["Eto"]
    x5 = tabelinha5[0].drop("Data", axis=1)
    y5 = df_patricia["Eto"]
    x6 = tabelinha6[0].drop("Data", axis=1)
    y6 = df_patricia["Eto"]
    x7 = tabelinha7[0].drop("Data", axis=1)
    y7 = df_patricia["Eto"]
    x8 = tabelinha8[0].drop("Data", axis=1)
    y8 = df_patricia["Eto"]
    x9 = tabelinha9[0].drop("Data", axis=1)
    y9 = df_patricia["Eto"]
    x10 = tabelinha10[0].drop("Data", axis=1)
    y10 = df_patricia["Eto"]
    x11 = tabelinha11[0].drop("Data", axis=1)
    y11 = df_patricia["Eto"]
    x12 = tabelinha12[0].drop("Data", axis=1)
    y12 = df_patricia["Eto"]
    x13 = tabelinha13[0].drop("Data", axis=1)
    y13 = df_patricia["Eto"]
    x14 = tabelinha14[0].drop("Data", axis=1)
    y14 = df_patricia["Eto"]
    x15 = tabelinha15[0].drop("Data", axis=1)
    y15 = df_patricia["Eto"]

    print("Separando o treino e o teste das tabelas...")
    print("TABELA 1")
    train_size1 = int((len(tabelinha1[0])-tabelinha1[1]) * 0.8)
    x1_train, x1_test = x1[tabelinha1[1]:train_size1], x1[train_size1:len(x1)]
    y1_train, y1_test = y1[tabelinha1[1]:train_size1], y1[train_size1:len(y1)]
    print("TABELA 2")
    train_size2 = int((len(tabelinha2[0])-tabelinha2[1]) * 0.8)
    x2_train, x2_test = x2[tabelinha2[1]:train_size2], x2[train_size2:len(x2)]
    y2_train, y2_test = y2[tabelinha2[1]:train_size2], y2[train_size2:len(y2)]
    print("TABELA 3")
    train_size3 = int((len(tabelinha3[0])-tabelinha3[1]) * 0.8)
    x3_train, x3_test = x3[tabelinha3[1]:train_size3], x3[train_size3:len(x3)]
    y3_train, y3_test = y3[tabelinha3[1]:train_size3], y3[train_size3:len(y3)]
    print("TABELA 4")
    train_size4 = int((len(tabelinha4[0])-tabelinha4[1]) * 0.8)
    x4_train, x4_test = x4[tabelinha4[1]:train_size4], x4[train_size4:len(x4)]
    y4_train, y4_test = y4[tabelinha4[1]:train_size4], y4[train_size4:len(y4)]
    print("TABELA 5")
    train_size5 = int((len(tabelinha5[0])-tabelinha5[1]) * 0.8)
    x5_train, x5_test = x5[tabelinha5[1]:train_size5], x5[train_size5:len(x5)]
    y5_train, y5_test = y5[tabelinha5[1]:train_size5], y5[train_size5:len(y5)]
    print("TABELA 6")
    train_size6 = int((len(tabelinha6[0])-tabelinha6[1]) * 0.8)
    x6_train, x6_test = x6[tabelinha6[1]:train_size6], x6[train_size6:len(x6)]
    y6_train, y6_test = y6[tabelinha6[1]:train_size6], y6[train_size6:len(y6)]
    print("TABELA 7")
    train_size7 = int((len(tabelinha7[0])-tabelinha7[1]) * 0.8)
    x7_train, x7_test = x7[tabelinha7[1]:train_size7], x7[train_size7:len(x7)]
    y7_train, y7_test = y7[tabelinha7[1]:train_size7], y7[train_size7:len(y7)]
    print("TABELA 8")
    train_size8 = int((len(tabelinha8[0])-tabelinha8[1]) * 0.8)
    x8_train, x8_test = x8[tabelinha8[1]:train_size8], x8[train_size8:len(x8)]
    y8_train, y8_test = y8[tabelinha8[1]:train_size8], y8[train_size8:len(y8)]
    print("TABELA 9")
    train_size9 = int((len(tabelinha9[0])-tabelinha9[1]) * 0.8)
    x9_train, x9_test = x9[tabelinha9[1]:train_size9], x9[train_size9:len(x9)]
    y9_train, y9_test = y9[tabelinha9[1]:train_size9], y9[train_size9:len(x9)]
    print("TABELA 10")
    train_size10 = int((len(tabelinha10[0])-tabelinha10[1]) * 0.8)
    x10_train, x10_test = x10[tabelinha10[1]:train_size10], x10[train_size10:len(x10)]
    y10_train, y10_test = y10[tabelinha10[1]:train_size10], y10[train_size10:len(y10)]
    print("TABELA 11")
    train_size11 = int((len(tabelinha11[0])-tabelinha11[1]) * 0.8)
    x11_train, x11_test = x11[tabelinha11[1]:train_size11], x11[train_size11:len(x11)]
    y11_train, y11_test = y11[tabelinha11[1]:train_size11], y11[train_size11:len(y11)]
    print("TABELA 12")
    train_size12 = int((len(tabelinha12[0])-tabelinha12[1]) * 0.8)
    x12_train, x12_test = x12[tabelinha12[1]:train_size12], x12[train_size10:len(x12)]
    y12_train, y12_test = y12[tabelinha12[1]:train_size12], y12[train_size10:len(y12)]
    print("TABELA 13")
    train_size13 = int((len(tabelinha13[0])-tabelinha13[1]) * 0.8)
    x13_train, x13_test = x13[tabelinha13[1]:train_size13], x13[train_size13:len(x13)]
    y13_train, y13_test = y13[tabelinha13[1]:train_size13], y13[train_size13:len(y13)]
    print("TABELA 14")
    train_size14 = int((len(tabelinha14[0])-tabelinha14[1]) * 0.8)
    x14_train, x14_test = x14[tabelinha14[1]:train_size14], x14[train_size14:len(x14)]
    y14_train, y14_test = y14[tabelinha14[1]:train_size14], y14[train_size14:len(y14)]
    print("TABELA 15")
    train_size15 = int((len(tabelinha14[0])-tabelinha15[1]) * 0.8)
    x15_train, x15_test = x15[tabelinha15[1]:train_size15], x15[train_size15:len(x15)]
    y15_train, y15_test = y15[tabelinha15[1]:train_size15], y15[train_size15:len(y15)]

    print("Montando modelos de Decision Tree Regressor...")
    # Fit regression model
    model_1 = DecisionTreeRegressor(max_depth = 4)
    model_2 = DecisionTreeRegressor(max_depth = 4)
    model_3 = DecisionTreeRegressor(max_depth = 4)
    model_4 = DecisionTreeRegressor(max_depth = 4)
    model_5 = DecisionTreeRegressor(max_depth = 4)
    model_6 = DecisionTreeRegressor(max_depth = 4)
    model_7 = DecisionTreeRegressor(max_depth = 2)
    model_8 = DecisionTreeRegressor(max_depth = 2)
    model_9 = DecisionTreeRegressor(max_depth = 2)
    model_10 = DecisionTreeRegressor(max_depth = 2)
    model_11 = DecisionTreeRegressor(max_depth = 2)
    model_12 = DecisionTreeRegressor(max_depth = 3)
    model_13 = DecisionTreeRegressor(max_depth = 3)
    model_14 = DecisionTreeRegressor(max_depth = 3)
    model_15 = DecisionTreeRegressor(max_depth = 3)
    print("Treinando modelos de Decision Tree Regressor criados...")
    model_1.fit(x1_train, y1_train)
    model_2.fit(x2_train, y2_train)
    model_3.fit(x3_train, y3_train)
    model_4.fit(x4_train, y4_train)
    model_5.fit(x5_train, y5_train)
    model_6.fit(x6_train, y6_train)
    model_7.fit(x7_train, y7_train)
    model_8.fit(x8_train, y8_train)
    model_9.fit(x9_train, y9_train)
    model_10.fit(x10_train, y10_train)
    model_11.fit(x11_train, y11_train)
    model_12.fit(x12_train, y12_train)
    model_13.fit(x13_train, y13_train)
    model_14.fit(x14_train, y14_train)
    model_15.fit(x15_train, y15_train)
    print("Predizendo modelos de Decision Tree Regressor...")
    # Predict
    y1_pred = model_1.predict(x1_test)
    y2_pred = model_2.predict(x2_test)
    y3_pred = model_3.predict(x3_test)
    y4_pred = model_4.predict(x4_test)
    y5_pred = model_5.predict(x5_test)
    y6_pred = model_6.predict(x6_test)
    y7_pred = model_7.predict(x7_test)
    y8_pred = model_8.predict(x8_test)
    y9_pred = model_9.predict(x9_test)
    y10_pred = model_10.predict(x10_test)
    y11_pred = model_11.predict(x11_test)
    y12_pred = model_12.predict(x12_test)
    y13_pred = model_13.predict(x13_test)
    y14_pred = model_14.predict(x14_test)
    y15_pred = model_15.predict(x15_test)
    print("Calculando o erro dos modelos...")
    tab_manual_erro = []
    tab_manual_erro

    tab_manual_erro.append(mean_squared_error(y1_test, y1_pred))
    tab_manual_erro.append(mean_squared_error(y2_test, y2_pred))
    tab_manual_erro.append(mean_squared_error(y3_test, y3_pred))
    tab_manual_erro.append(mean_squared_error(y4_test, y4_pred))
    tab_manual_erro.append(mean_squared_error(y5_test, y5_pred))
    tab_manual_erro.append(mean_squared_error(y6_test, y6_pred))
    tab_manual_erro.append(mean_squared_error(y7_test, y7_pred))
    tab_manual_erro.append(mean_squared_error(y8_test, y8_pred))
    tab_manual_erro.append(mean_squared_error(y9_test, y9_pred))
    tab_manual_erro.append(mean_squared_error(y10_test,y10_pred))
    tab_manual_erro.append(mean_squared_error(y11_test,y11_pred))
    tab_manual_erro.append(mean_squared_error(y12_test,y12_pred))
    tab_manual_erro.append(mean_squared_error(y13_test,y13_pred))
    tab_manual_erro.append(mean_squared_error(y14_test,y14_pred))
    tab_manual_erro.append(mean_squared_error(y15_test,y15_pred))
    
    r =np.array(tab_manual_erro)
    tab_manual_erro =pow(r,0.5)


    lista_tabelas =[tabelinha1,
                    tabelinha2,
                    tabelinha3,
                    tabelinha4,
                    tabelinha5,
                    tabelinha6,
                    tabelinha7,
                    tabelinha8,
                    tabelinha9,
                    tabelinha10,
                    tabelinha11,
                    tabelinha12,
                    tabelinha14,
                    tabelinha15]

    lista_colunas=["lista","lags","Eto","lags_eto"]
    print("Criando dataFrame com os erros...")
    tb = pd.DataFrame(columns=lista_colunas)
    tabelinha1[2]

    
    tb['lista']=[tabelinha1[2],tabelinha2[2],tabelinha3[2],tabelinha4[2],tabelinha5[2],tabelinha6[2],tabelinha7[2],tabelinha8[2],tabelinha9[2],tabelinha10[2],tabelinha11[2],tabelinha12[2],tabelinha13[2], tabelinha14[2],tabelinha15[2]]
    tb['lags']=[tabelinha1[3],tabelinha2[3],tabelinha3[3],tabelinha4[3],tabelinha5[3],tabelinha6[3],tabelinha7[3],tabelinha8[3],tabelinha9[3],tabelinha10[3],tabelinha11[3],tabelinha12[3],tabelinha13[3], tabelinha14[3],tabelinha15[3]]
    tb['lags_eto']=[tabelinha1[5],tabelinha2[5],tabelinha3[5],tabelinha4[5],tabelinha5[5],tabelinha6[5],tabelinha7[5],tabelinha8[5],tabelinha9[5],tabelinha10[5],tabelinha11[5],tabelinha12[5],tabelinha13[5], tabelinha14[5],tabelinha15[5]]
    tb["erro_rmse"]=tab_manual_erro
    print("Exportando dados lags e rmse ...")
    tb.to_csv('./Dados/erro_rmse.csv')
    print("Concluido !")
    print(tb)
    return tb




def run_arimaDay(series, steps_ahead, configuracao):
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #Valores da série
  values = series.values

  #ARIMA
  start_fit = time.time()
  mod = sm.tsa.statespace.SARIMAX(values, order=configuracao)
  res = mod.fit(disp=False)
  t_fit = time.time() - start_fit
  
  start_fcast = time.time() 
  forecast = res.forecast(steps=steps_ahead)
  t_fcast = time.time() - start_fcast 
  
  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i].strftime('%d/%m/%Y '),0])
    else:
      result.append([date_list[i].strftime('%d/%m/%Y '),round((forecast[i]),3)])

  return result, t_fit, t_fcast

def run_sarimaDay(series, steps_ahead, config_ordem, config_sazonal):
  result = []
  
  #Lista de data+hora que será previsto
  begin = series.index.max() + timedelta(days=0)
  date_list = [begin + timedelta(days=x) for x in range(1,steps_ahead+1)]
  
  #Valores da série
  values = series.values

  #ARIMA
  mod = sm.tsa.statespace.SARIMAX(values, order=config_ordem, seasonal_order=config_sazonal)
  res = mod.fit(disp=False)
  forecast = res.forecast(steps=steps_ahead)

  #Resultado no formato para ser exibido no gráfico
  for i in range(steps_ahead):
    if forecast[i] < 0: 
      result.append([date_list[i],0])
    else:
      result.append([date_list[i],round((forecast[i]),3)])

  return result


def arvore(df,lista, lags, Eto, lags_eto):
  tabela = get_x2(df, lista,lags,Eto,lags_eto)
  x1 = tabela[0].drop("Data", axis=1)
  y1 = df["Eto"]
  train_size = int(((len(tabela[0])) -tabela[1]) * 0.8)
  x1_train, x1_test = x1[tabela[1]:train_size], x1[train_size:len(x1)]
  y1_train, y1_test = y1[tabela[1]:train_size], y1[train_size:len(y1)]
  model = DecisionTreeRegressor(max_depth = 3)
  model.fit(x1_train, y1_train)
  print("Aqui")
  y1_pred = model.predict(x1_test)
  mse = mean_squared_error(y1_test, y1_pred)
  rmse = math.sqrt(mse)
    
  return lista,lags,Eto,lags_eto,rmse

def arvores(df,arvore_parametros):
  lista_colunas=["lista","lista_lags","Eto","lags_eto","rmse"]
  tb = pd.DataFrame(columns=lista_colunas)
  print("oi")
  for x in range(len(arvore_parametros)):
    a=arvore(df,arvore_parametros[x][0],arvore_parametros[x][1],arvore_parametros[x][2],arvore_parametros[x][3])
 
    tb.loc[x,'lista']=a[0]
    tb.loc[x,'lista_lags']=a[1]
    tb.loc[x,'Eto']=a[2]
    tb.loc[x,'lags_eto']=a[3]
    tb.loc[x,"rmse"]=a[4]

  print(tb)

  return tb

