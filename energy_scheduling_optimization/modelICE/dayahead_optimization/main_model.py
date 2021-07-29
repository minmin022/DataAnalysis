# 日前设备出力优化
import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank
from modelICE.model import GasTurbine as GT
from modelICE.model import Magnetic_EleChiller as Mag_Elechiller
from modelICE.model import AbsorptionChiller as Abchiller
from modelICE.model import VariaFrequency_EleChiller as HeatPump    #用vf_elechiller 的运行曲线模拟heatpump
from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import demand
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 负荷预测
dayahead_ele_load = demand.dayahead_ele_load
dayahead_cold_load = demand.dayahead_cold_load

# 7个设备出力
"""
gasturbine, absorptionchiller,elechiller,heatpump,grid,
boiler(制热 制冷过程不考虑),watertank,battery
优化7个设备24个小时的出力情况
"""
total_cost_m_t = np.zeros(24, dtype=np.int32)
total_gasturbine_m_t = np.zeros(24, dtype=np.int32)
total_absorpchiller_m_t = np.zeros(24, dtype=np.int32)
total_elechiller_m_t = np.zeros(24, dtype=np.int32)
total_heatpump_m_t = np.zeros(24, dtype=np.int32)
total_grid_m_t = np.zeros(24, dtype=np.int32)
total_watertank_m_t = np.zeros(24, dtype=np.int32)
total_battery_m_t = np.zeros(24, dtype=np.int32)
n = 7  #7个设备
M = 24 #优化时域为24h
# 运行成本

# 构建目标方程
def cost_min(x):  # x为 n*M列表
    # def demo_func(x):
    global dayahead_gasturbine_ele,dayahead_gasturbine_heat,dayahead_gasturbine_heat_in, \
        dayahead_heatpump, dayahead_elechiller, dayahead_absorpchiller, dayahead_grid,\
        dayahead_watertank, dayahead_battery, cost_min
    # cost_min = 0
    # #     dayahead_gasturbine_t_i = x[0]
    # #     dayahead_absorpchiller_t_i = x[1]
    # #     dayahead_elechiller_t_i = x[2]
    # #     dayahead_heatpump_t_i = x[3]
    # #     dayahead_grid_t_i = x[4]
    dayahead_gasturbine_ele = []  # 0
    dayahead_gasturbine_heat = []
    dayahead_gasturbine_heat_in = []
    dayahead_absorpchiller = []  # 1
    dayahead_elechiller = []  # 2
    dayahead_heatpump = []  # 3
    dayahead_grid = []  # 4
    dayahead_watertank = []  # 5  #watertank 为正 表示释能； 为负 表示蓄能
    dayahead_battery = []  # 6

    for m in range(len(x)):  #7*24
        if m % n == 0:
            dayahead_gasturbine_ele.append(x[m])  # 发电量
            # dayahead_gasturbine[]
        elif m % n == 1:
            dayahead_absorpchiller.append(x[m])
        elif m % n == 2:
            dayahead_elechiller.append(x[m])
        elif m % n == 3:
            dayahead_heatpump.append(x[m])
        elif m % n == 4:
            dayahead_grid.append(x[m])
        elif m % n == 5:
            dayahead_watertank.append(x[m])
        elif m % n == 6:
            dayahead_battery.append(x[m])


    # 燃料成本
    for gt_ele in dayahead_gasturbine_ele:
        gt_heat = GT.GasTurbine(1500).get_heat_in(gt_ele)[1]  # 假设GT额定功率为1500kw
        dayahead_gasturbine_heat.append(gt_heat)
        gt_heat_in = GT.GasTurbine(1500).get_heat_in(gt_ele)[0]
        dayahead_gasturbine_heat_in.append(gt_heat_in)
    fuel_consume = sum(dayahead_gasturbine_heat_in) * eco_pr.delttime_dayin / eco_pr.heatvalue  # kwh/(Kwh# /m³) = m³
    fuel_cost = fuel_consume * eco_pr.fuel_price  # 元

    # 运维成本

    maintain_cost = eco_pr.cost_gasturbine * eco_pr.delttime_dayin * sum(dayahead_gasturbine_ele) + \
                    eco_pr.cost_absorptionchiller * eco_pr.delttime_dayin * sum(dayahead_absorpchiller) + \
                    eco_pr.cost_elechiller * eco_pr.delttime_dayin * sum(dayahead_elechiller) + \
                    eco_pr.cost_heatpump * eco_pr.delttime_dayin * sum(dayahead_heatpump) + \
                    eco_pr.cost_watertank * eco_pr.delttime_dayin * sum(abs(x) for x in dayahead_watertank) + \
                    eco_pr.cost_battery * eco_pr.delttime_dayin * sum(abs(y) for y in dayahead_battery)

    """
    cost_gasturbine
    cost_heatpump
    cost_chiller
    cost_coldtank
    加上watertank，battery的运维成本
    由于有正负值（蓄能/放能），因此选择绝对值
    """


    # 购电成本
    grid_cost = 0
    for k in range(M):
        grid_cost += eco_pr.get_ele_price(k) * eco_pr.delttime_dayin * dayahead_grid[k]  # 改 电价会发生变化

    # 总成本
    total_cost_min = fuel_cost + maintain_cost + grid_cost
    return total_cost_min


def penalty_cold_balance():
    # 冷负荷平衡
    p_cold_balance = 0
    for k in range(M):
        p_cold_balance += (dayahead_absorpchiller[k] + dayahead_elechiller[k] + dayahead_heatpump[k] + dayahead_watertank[k]
                           - dayahead_cold_load[k]) ** 2
    return p_cold_balance


def penalty_ele_balance():
    # 电负荷平衡
    p_ele_balance = 0
    for k in range(M):
        p_ele_balance += (dayahead_gasturbine_ele[k] + dayahead_grid[k] + dayahead_battery[k]- Mag_Elechiller.Magnetic_EleChiller2.get_ele_in(
            dayahead_elechiller[k])
                          - HeatPump.VariaFrequency_EleChiller2.get_ele_in(dayahead_heatpump[k]) - dayahead_ele_load[k]) ** 2
    return p_ele_balance

"""
用vf_elechiller性能曲线模拟heatpump，heatpump模型需要再完善
"""

def demo_func(x):
    m = 10000
    return cost_min(x) + m * (penalty_cold_balance() + penalty_ele_balance())



# 约束
constraint_ueq_list = []
#gasturbine
def constraint_gasturbine(x):
    cost_min(x)
    constraint_gasturbine = []
    for k in range(M):
        # 爬坡约束
        constraint_upper = abs(dayahead_gasturbine_ele[k]-dayahead_gasturbine_ele[k-1])-60
        constraint_gasturbine.append(constraint_upper)
    return constraint_gasturbine

#absorptionchiller
def constraint_absorptionchiller(x):
    cost_min(x)
    constraint_absorptionchiller = []
    for k in range(M):
        #制冷能力上限
        constraint_capacity = Abchiller.AbsorptionChiller2460.get_ele_in(dayahead_absorpchiller[k]) - dayahead_gasturbine_heat[k]
        # 吸收式制冷heat in - gasturbine heat out
        constraint_absorptionchiller.append(constraint_capacity)
    return constraint_absorptionchiller


constraint_ueq_list.append(lambda x:constraint_gasturbine(x))
constraint_ueq_list.append(lambda x:constraint_absorptionchiller(x))
# constraint_ueq_list里面有list
# pso优化
pso = PSO(func=demo_func, n_dim=n*M, pop=2000, max_iter=80, lb=[0,0,0,0,0,-400,-400]*M, ub=[1500,1500,800,500,2000,400,400]*M,
                  constraint_ueq=tuple(constraint_ueq_list), verbose=1)     #[gasturbine,absorpchiller,elechiller,heatpump,grid,watertank,battery]
        # 假设watertank 释能和蓄能的max power 相同；均为[0,1500]

pso.record_mode = True
pso.run()
print(len(pso.gbest_x))
print(pso.gbest_x)
print(pso.gbest_y)
