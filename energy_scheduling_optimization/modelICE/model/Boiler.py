#修改！
from modelICE.Parameters import Parameters2 as Pr

# 燃气锅炉
n1 = 2  #1.4MW两台
nominal_Boiler1 = 2800  # kw 额定制热量
effi_Boiler1 = \
   {0.1: 0.902, 0.2:0.94, 0.3:0.976, 0.4:1.012, 0.5:1.049, 0.6:1.05, 0.7: 1.05, 0.8:1.05, 0.9:1.046, 1: 1.042}  # pl:effi
n2 = 1 #0.7MW一台
nominal_Boiler2 = 700  # kw 额定制热量
effi_Boiler2 = \
    {0.1: 0.902, 0.2: 0.96, 0.3: 1.016, 0.4: 1.038, 0.5: 1.049, 0.6: 1.05, 0.7: 1.05, 0.8: 1.05, 0.9: 1.046,
     1: 1.042}  # pl:effi

heatvalue = 38931  # 天然气热值 KJ/m³
# 35.588MJ/Nm³

class Boiler(Pr.Heat_cold_generator):
    def __init__(self, nominal, performance):
        super().__init__(nominal, performance)

    # def get_heat_in(self,heat_out):
    #     pl = heat_out / self.nominal
    #     effi = float(self.get_COP(pl))
    #     Boiler_heat_in = heat_out / effi
    #     return Boiler_heat_in
    #
    # def get_heat_out(self,heat_in):
    #     pl = self.get_pl(heat_in)
    #     Boiler_heat_out = self.nominal * pl
    #     return Boiler_heat_out

Boiler1400 = Boiler(nominal_Boiler1, effi_Boiler1)

if __name__ == '__main__':
    Boiler1 = Boiler(nominal_Boiler1,effi_Boiler1)
    print(Boiler1.get_cold_out(1340))  #输入具体的heat_in数据
    print(Boiler1.get_ele_in(1400))




# 锅炉效率是指锅炉使用期间、蒸汽带走的热量与燃料的低热值之比