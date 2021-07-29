from modelICE.Parameters import Parameters2 as Pr

#data
effi_EleStorage_dis = 0.95  # 蓄电池放电效率，待定
effi_EleStorage_chr = 0.95  # 蓄电池充电效率，待定

class EleStorage:
    def __init__(self,effi_dis,effi_chr):
        self.effi_dis = effi_dis
        self.effi_chr = effi_chr  # 充放电功率是一定的吗？充点电效率会随充放电功率波动吗？

# 充电 ele_chr_dis>0 ; 放电 ele_chr_dis<0

    def get_EleStorage(self,ele_chr_dis,ele_stor=0):
        global elestorage
        # 可能要加判断， if ele_dis/chr 在一定范围，不变；不在一定范围，取端值
        if ele_chr_dis >= 0:  #放电
            # u_chr_dis = -1
            elestorage = ele_stor - Pr.delttime * (ele_chr_dis * self.effi_chr)
        elif ele_chr_dis < 0:  #蓄电
            # u_chr_dis = -1
            elestorage = ele_stor - Pr.delttime * (ele_chr_dis * self.effi_dis)

        return elestorage


EleStorage1 = EleStorage(effi_EleStorage_dis,effi_EleStorage_chr)

















