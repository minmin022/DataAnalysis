
from modelICE.Parameters import Parameters2 as Pr

#吸收式制冷机组
nominal_AbsorptionChiller = 1200  # kw 额定制冷量  要改！！
COP_AbsorptionChiller = \
        {0.1:1.17,0.2: 1.2, 0.4: 1.22, 0.6: 1.21, 0.8: 1.18, 1: 1.15}



class AbsorptionChiller(Pr.Heat_cold_generator):
    def __init__(self,nominal,performance):
        super().__init__(nominal, performance)

    # def get_heat_in(self,cold_out):
    #     pl = cold_out / self.nominal
    #     COP = self.get_COP(pl)
    #     absorptionChiller_heat_in = cold_out / COP
    #     return absorptionChiller_heat_in
    #
    # def get_cold_out(self,heat_in):
    #     pl = self.get_pl(heat_in)
    #     absorptionChiller_cold_out = self.nominal * pl
    #     return absorptionChiller_cold_out

AbsorptionChiller2460 = AbsorptionChiller(nominal_AbsorptionChiller,COP_AbsorptionChiller)
AbsorptionChiller1200 = AbsorptionChiller(1200,COP_AbsorptionChiller) #DEMO用

if __name__ == '__main__':
    AbsorptionChiller1 = AbsorptionChiller(nominal_AbsorptionChiller,COP_AbsorptionChiller)
    print(AbsorptionChiller1.get_cold_out(230))  #输入具体的heat_in数据
    print(AbsorptionChiller1.get_ele_in(67.33))


