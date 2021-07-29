
from modelICE.Parameters import Parameters2 as Pr

# 热泵
nominal_heatpump = 281.3  # kw 热泵额定制冷量
COP_heatpump = \
    { 0.3: 3.2832,0.35:3.6551, 0.4: 3.9944,0.45:4.3045, 0.5: 4.5903, 0.55:4.8539, 0.6: 4.9531, 0.7: 5.0524, 0.8: 5.0411, 0.9: 4.9273,
     1: 3.6379}  # pl：COP

class Heatpump(Pr.Heat_cold_generator):
    def __init__(self, nominal, performance):
        super().__init__(nominal, performance)


    # def get_ele_in(self,cold_out):
    #     pl = cold_out / self.nominal
    #     COP = self.get_COP(pl)
    #     Magnetic_EleChiller_ele_in = cold_out / COP
    #     return Magnetic_EleChiller_ele_in
    #
    # def get_cold_out(self,ele_in):
    #     pl = self.get_pl(ele_in)
    #     Magnetic_EleChiller_cold_out = self.nominal * pl
    #     return Magnetic_EleChiller_cold_out

heatpump1 = Heatpump(nominal_heatpump,COP_heatpump)


if __name__ == '__main__':
    heatpump1 = Heatpump(nominal_heatpump,COP_heatpump)
    print(heatpump1.get_cold_out(51.3))  #输入具体的ele_in数据
    print(heatpump1.get_ele_in(253.1))


