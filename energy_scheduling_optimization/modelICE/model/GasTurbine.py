from modelICE.Parameters import Parameters2 as Pr
import math
# e = 2.718281828459
# el_nominal = 330  # kw Jenbacher Type J208
# heat_nominal = 371   # nominal el
# el_effi = 38.8    # nominal el
# heat_effi = 43.6     # nominal el
# 热电比是一定的吗？ el_effi.可用el_pl拟合，heat_effi，heat_pl 怎么确定

el_nominal = 1200 #kw
# el_effi_nominal = 0.04236 * math.log(el_nominal) + 0.115
# total_effi = 0.02303 * math.log(el_nominal) + 0.703
class GasTurbine():
    def __init__(self, el_nominal):
        self.el_nominal = el_nominal
        self.el_effi_nominal = 0.04236 * math.log(el_nominal) + 0.115
        self.total_effi = 0.02303 * math.log(el_nominal) + 0.703

    # heat_in
    # heat_out
    def get_heat_in(self,el_out):
        pl = el_out / self.el_nominal
        a = 1.334
        b = -3.208
        c = 2.605
        d = 0.268
        el_effi = (a * pow(pl, 3) + b * pow(pl, 2) + c * pl + d) * self.el_effi_nominal
        # if el_effi is 0:
        #     el_effi = 0.1
        heat_effi = self.total_effi - el_effi
        gasturbine_heat_in = el_out / el_effi
        gasturbine_heat_out = gasturbine_heat_in * heat_effi
        return gasturbine_heat_in, gasturbine_heat_out

    # ele_out
    # def get_el_out(self,heat_in):

GasTurbine1 = GasTurbine(el_nominal )