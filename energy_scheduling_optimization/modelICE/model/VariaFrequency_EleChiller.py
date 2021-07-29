
from modelICE.Parameters import Parameters2 as Pr

# 变频离心制冷机组
nominal_VariaFrequency_EleChiller = 4920  # kw 额定制冷量  2460*2
COP_VariaFrequency_EleChiller = \
    {0:1, 0.1: 1.257, 0.2: 2.514, 0.3: 3.771, 0.4: 4.935, 0.5: 5.726, 0.6: 5.88, 0.7: 5.797, 0.8: 5.673, 0.9: 5.539,
     1: 5.312}  # 工况1  pl:COP   Loadline 4


class VariaFrequency_EleChiller(Pr.Heat_cold_generator):
    def __init__(self, nominal, performance):
        super().__init__(nominal, performance)

    # def get_ele_in(self,cold_out):
    #     pl = cold_out / self.nominal
    #     COP = self.get_COP(pl)
    #     VariaFrequency_EleChiller_ele_in = cold_out / COP
    #     return VariaFrequency_EleChiller_ele_in
    #
    # def get_cold_out(self,ele_in):
    #     pl = self.get_pl(ele_in)
    #     VariaFrequency_EleChiller_cold_out = self.nominal * pl
    #     return VariaFrequency_EleChiller_cold_out

VariaFrequency_EleChiller1 = VariaFrequency_EleChiller(nominal_VariaFrequency_EleChiller,COP_VariaFrequency_EleChiller)

VariaFrequency_EleChiller2 = VariaFrequency_EleChiller(100,COP_VariaFrequency_EleChiller) #DEMO


if __name__ == '__main__':
    VariaFrequency_EleChiller1 = VariaFrequency_EleChiller(nominal_VariaFrequency_EleChiller,COP_VariaFrequency_EleChiller)
    print(VariaFrequency_EleChiller1.get_cold_out(463.1))  #输入具体的ele_in数据
    print(VariaFrequency_EleChiller1.get_ele_in(1722))


