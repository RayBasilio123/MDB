order_rad = (24,0,24) #graficos
order_eto1 = (6, 0, 41) #graficos
order_eto2 = (5, 1, 2) #graficos
order_eto2_sazonal = (1,0,1,3)

order_eto_ARIMA = (1,1,3)
order_eto_SARIMA=(6,1,1)
order_eto_sazonal_SARIMA = (2,1,0,12)

result_ven_1 = run_arima(series_ventovel,24,order_ven)
result_ven_1