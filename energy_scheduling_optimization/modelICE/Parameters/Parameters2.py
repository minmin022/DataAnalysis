import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc

class Heat_cold_generator:

    def __init__(self,nominal,performance):
        self.nominal = nominal
        self.performance = performance

        self.x = np.array(tuple(performance.keys()))
        self.z = np.array(tuple(performance.values()))
        self.y = self.x / self.z

        self.a = np.polyfit(self.x, self.y, 5)  # 用2次多项式拟合x，y数组
        # print(a)
        self.b = np.poly1d(self.a)  # 拟合完之后用这个函数来生成多项式对象
        # print(b)
        self.c = self.b(self.x)  # 生成多项式对象之后，就是获取x在这个多项式处的值
        # plt.scatter(x, y, marker='o', label='original datas')  # 对原始数据画散点图
        # plt.plot(x, c, ls='--', c='red', label='fitting with second-degree polynomial')  # 对拟合之后的数据，也就是x，c数组画图
        # plt.legend()
        # plt.show()

    def get_ele_in(self,cold_out):
        ele_in = np.polyval(self.b, cold_out/self.nominal) * self.nominal
        return ele_in

    def get_cold_out(self,ele_in):
        pl = inversefunc(self.b,y_values=ele_in/self.nominal)
        # print(pl)
        cold_out = self.nominal * pl
        # print(cold_out)
        return cold_out

 #
 # #  制冷机模块
 # #  制冷机负荷百分比 vs COP
 #    # 已知cooling求输入能量
 #    def get_COP(self,pl_run):
 #        prePl = 999
 #        for Pl in self.performance.keys():
 #            if prePl <= pl_run and pl_run <= Pl:
 #                x1 = prePl
 #                y1 = self.performance[prePl]
 #                x2 = Pl
 #                y2 = self.performance[Pl]
 #                cop_run = y1 + ((y2 - y1) / (x2 - x1)) * (pl_run - x1)
 #                return cop_run
 #            prePl = Pl
 #        return 1
 #
 #    #已知输入能量（ele,heat）求cooling
 #    def get_pl(self, ele_in):
 #        keys = list(self.performance.keys())
 #        values = list(self.performance.values())
 #        slope = {}
 #        for i in range(0, len(keys) - 1):
 #            n = keys[i]
 #            slope[n] = round((values[i + 1] - values[i]) / (keys[i + 1] - keys[i]), 2)
 #        num = len(keys)
 #        for i in range(1, num, 1):
 #            pl_0 = i / num
 #            pl_run = (self.performance[pl_0] - slope[pl_0] * pl_0) / (self.nominal / ele_in - slope[pl_0])
 #            if pl_0 <= pl_run < pl_0 + 1/num:
 #                return pl_run
 #        return 1

#
# #吸收式制冷机组  CHP
# nominal_AbsorptionChiller = 2460  # kw 额定制冷量  要改！！
# COP_AbsorptionChiller = \
#     {0.2: 0.5, 0.4: 0.8, 0.6: 1.1, 0.8: 1.3, 1: 1}   #要改！！
#
# # 变频离心制冷机组
# nominal_VariaFrequency_EleChiller = 2460  #kw 额定制冷量
# COP_VariaFrequency_EleChiller = \
#     {0.1: 1.257, 0.2: 2.514, 0.3: 3.771, 0.4: 4.935, 0.5: 5.726, 0.6: 5.88, 0.7: 5.797, 0.8: 5.673, 0.9: 5.539,
#      1: 5.312}  # 工况1  pl:COP
# P_VariaFrequency_EleChiller_max = 2460
# P_VariaFrequency_EleChiller_min = 1200
#
# # 磁悬浮电制冷机组
# nominal_Magenic_EleChiller = 120 #kw 额定制冷量
# COP_Magnetic_EleChiller =  \
#     {0.1:5.706, 0.2:9.166, 0.3:11.28, 0.4:11.95, 0.5:12.05, 0.6:10.11, 0.7:8.506, 0.8:7.319, 0.9:6.579, 1:5.72} #pl：COP
# P_Magnetic_EleChiller_max = 120
# P_Magnetic_EleChiller_min = 60
#
# # 热泵
# effi_HeatPump = 0.5 #效率
#
# # 蓄水池
#
# #燃气锅炉
# P_boiler_max = 700 # 改！ 运行上限
# P_boiler_min = 300  # 改！ 运行下限
#
# # 燃气轮机
# el_nominal = 600
# P_gasturbine_max = 600  # 改！ 运行上限
# P_gasturbine_min = 300  # 改！ 运行下限
# P_gasturbine_up = 200 # 改！ 机组爬坡率限制最大上升功率
# P_gasturbine_down = 200 # 改！ 机组爬坡率限制最大上升功率
#
# # 蓄电池
# P_elestorage_chr_max = 400
# P_elestorage_chr_min = 100
# P_elestorage_dis_max = 400
# P_elestorage_dis_min = 100



# heatvalue = 38931  # 天然气热值 KJ/m³
# 35.588MJ/Nm³
delttime = 1  # h 时间间隔



