# 以下为设备运行成本参数：
# cost_GasTurbine_per_kw = 3000  # 元/kW
# cost_Boiler_per_kw = 300
# cost_AbsorptionChiller_per_kw = 1200
# cost_Magnetic_EleChiller_per_kw = 1200
# cost_VariaFrequency_Elechiller_per_kw = 1200
# cost_HeatPump_per_kw = 970
# cost_Watertank_per_kw = 230
# cost_EleStorage_per_kw = 100

cost_gasturbine = 0.03 # 元/kwh
cost_absorptionchiller = 0.025
cost_elechiller = 0.01
cost_heatpump = 0.02
cost_boiler = 0.02  #余热锅炉
cost_watertank = 0.02 # 元/kwh
cost_battery = 0.02
# 设备单位运行成本


# fuel 天然气参数

heatvalue = 38931/3600  # 天然气热值 KWh /m³
# 35.588MJ/Nm³
fuel_price = 3.46/heatvalue #元/立方  #0.32元/kwh
#fuel_price = 0.348 #元/kwh
delttime_dayin = 1  # h 时间间隔


# 分时电价
ele_high_price = 1.1636   # 假设电压等级为1-10kv kwh
# ele_high_price = 5.1636   # 假设电压等级为1-10kv kwh
ele_medium_price = 0.8656
ele_low_price = 0.3536
# ele_low_price = 0.0001
def get_ele_price(t):
    global ele_price
    if 0 <= t < 8 or 11 <= t < 13 or 22 <= t < 24:
        ele_price = ele_low_price
    if 8 <= t < 11 or 13 <= t < 19 or 21 <= t < 22:
        ele_price = ele_medium_price
    if 19 <= t < 21:
        ele_price = ele_high_price
    return ele_price














































