
from modelICE.Parameters import Parameters2 as Pr

# 磁悬浮电制冷机组
nominal_Magenic_EleChiller = 120  # kw 热泵额定制冷量
COP_Magnetic_EleChiller = \
    {0:5,0.1: 5.706, 0.2: 9.166, 0.3: 11.28, 0.4: 11.95, 0.5: 12.05, 0.6: 10.11, 0.7: 8.506, 0.8: 7.319, 0.9: 6.579,
     1: 5.72}  # pl：COP

class Magnetic_EleChiller(Pr.Heat_cold_generator):
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

Magnetic_EleChiller1 = Magnetic_EleChiller(nominal_Magenic_EleChiller,COP_Magnetic_EleChiller)

Magnetic_EleChiller2 = Magnetic_EleChiller(600,COP_Magnetic_EleChiller)  #DEMO用
if __name__ == '__main__':
    Magnetic_EleChiller1 = Magnetic_EleChiller(nominal_Magenic_EleChiller,COP_Magnetic_EleChiller)
    print(Magnetic_EleChiller1.get_cold_out(20.97))  #输入具体的ele_in数据
    print(Magnetic_EleChiller1.get_ele_in(120))


