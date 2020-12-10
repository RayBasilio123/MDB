data_Patricia_Eto = pd.read_csv('C:/Users/Ray/Documents/2020_Ic/Dados/ETo_setelagoas.csv') 
data_Patricia= pd.read_csv('C:/Users/Ray/Documents/2020_Ic/Dados/variaveis_setelagoas.csv')
data_Patricia["Eto"] = data_Patricia_Eto["Eto"]

print("Exportando dataFrameDay ...")
dataFrameDay.to_csv('./Dados/data_Patricia.csv')
print("Concluido !")
print(tabelinhaP[0])

# atributeP= [ "Tmax","Tmean","I","UR","V","Tmin"]