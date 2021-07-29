from modelICE.Parameters import Parameters2 as Pr
from modelICE.Parameters import economy_parameters as eco_pr
#data
effi_watertank_out = 0.95  # 蓄水池释能效率，待定
effi_watertank_in = 0.95  # 蓄水池储能效率，待定

# 先只考虑冷能
class WaterTank:
    def __init__(self,effi_out,effi_in):
        self.effi_out = effi_out
        self.effi_in = effi_in


    def get_ColdStorage(self,cold_in_out,watertank_storage_t0=0):
        # global coldstorage

        if cold_in_out >= 0: #释能
            #u_in_out = -1
            coldstorage = watertank_storage_t0 - eco_pr.delttime_dayin * (cold_in_out * self.effi_in)
            #coldstorage = WaterTank.get_ColdStorage(self,x,t-1) + Pr.delttime * ( u_in_out * cold_in_out * self.effi_in)
        else: #蓄能
           # u_in_out = -1
            coldstorage = watertank_storage_t0 - eco_pr.delttime_dayin * (cold_in_out * self.effi_out)
            #coldstorage = WaterTank.get_ColdStorage(self,x,t-1) + Pr.delttime * (u_in_out * cold_in_out * self.effi_out)
        return coldstorage


WaterTank1 = WaterTank(effi_watertank_out, effi_watertank_in)

















