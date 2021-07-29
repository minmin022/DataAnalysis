# 日前设备出力优化
import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank
from modelICE.model import GasTurbine as GT
# from modelICE.model import Magnetic_EleChiller as Mag_Elechiller
from modelICE.model import VariaFrequency_EleChiller as Var_Elechiller
from modelICE.model import AbsorptionChiller as Abchiller
from modelICE.model import HeatPump
from modelICE.model import WaterTank as WaterTank
from modelICE.model import EleStorage as battery
from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import demand
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 负荷预测
dayahead_ele_load = demand.dayahead_ele_load * 2
dayahead_cold_load = demand.dayahead_cold_load * 2

# 7个设备出力
"""
gasturbine, absorptionchiller,elechiller,heatpump,grid,
boiler(制热 制冷过程不考虑),watertank,battery
优化7个设备24个小时的出力情况
"""
# total_cost_m_t = np.zeros(24, dtype=np.int32)
# total_gasturbine_m_t = np.zeros(24, dtype=np.int32)
# total_absorpchiller_m_t = np.zeros(24, dtype=np.int32)
# total_elechiller_m_t = np.zeros(24, dtype=np.int32)
# total_heatpump_m_t = np.zeros(24, dtype=np.int32)
# total_grid_m_t = np.zeros(24, dtype=np.int32)
# total_watertank_m_t = np.zeros(24, dtype=np.int32)
# total_battery_m_t = np.zeros(24, dtype=np.int32)


n = 7  #7个设备
M = 24 #优化时域为24h
# 运行成本

def vari_sepa(x):
    dayahead_gasturbine_ele = np.zeros(M, dtype=np.int32)  # 0
    dayahead_absorpchiller = np.zeros(M, dtype=np.int32)  # 1
    dayahead_elechiller = np.zeros(M, dtype=np.int32)  # 2
    dayahead_heatpump = np.zeros(M, dtype=np.int32)  # 3
    dayahead_grid = np.zeros(M, dtype=np.int32)  # 4
    dayahead_watertank = np.zeros(M, dtype=np.int32)  # 5  #watertank 为正 表示释能； 为负 表示蓄能
    dayahead_battery = np.zeros(M, dtype=np.int32)  # 6
    for m in range(len(x)):  #7*24
        dayahead_gasturbine_ele = x[0::7]  # 发电量
        dayahead_absorpchiller  = x[1::7]
        dayahead_elechiller     = x[2::7]
        dayahead_heatpump       = x[3::7]
        dayahead_grid           = x[4::7]
        dayahead_watertank      = x[5::7]
        dayahead_battery        = x[6::7]

    return dayahead_gasturbine_ele,dayahead_absorpchiller,dayahead_elechiller,dayahead_heatpump,\
           dayahead_grid,dayahead_watertank,dayahead_battery

# 构建目标方程
def cost_min(x):  # x为 n*M列表
    (dayahead_gasturbine_ele,dayahead_absorpchiller,dayahead_elechiller,
     dayahead_heatpump,dayahead_grid,dayahead_watertank,dayahead_battery) = vari_sepa(x)

    dayahead_gasturbine_heat = np.zeros(M, dtype=np.int32)
    dayahead_gasturbine_heat_in = np.zeros(M, dtype=np.int32)
    for u in range(len(dayahead_gasturbine_ele)):
        gt_ele = dayahead_gasturbine_ele[u]
        gt_heat = GT.GasTurbine1.get_heat_in(gt_ele)[1]
        dayahead_gasturbine_heat[u] = gt_heat
        gt_heat_in = GT.GasTurbine1.get_heat_in(gt_ele)[0]
        dayahead_gasturbine_heat_in[u] = gt_heat_in
    # 燃料成本

    fuel_consume = sum(dayahead_gasturbine_heat_in) * eco_pr.delttime_dayin
    fuel_cost = fuel_consume * eco_pr.fuel_price  # 元

    # 运维成本

    maintain_cost = eco_pr.cost_gasturbine * eco_pr.delttime_dayin * sum(dayahead_gasturbine_ele) + \
                    eco_pr.cost_absorptionchiller * eco_pr.delttime_dayin * sum(dayahead_absorpchiller) + \
                    eco_pr.cost_elechiller * eco_pr.delttime_dayin * sum(dayahead_elechiller) + \
                    eco_pr.cost_heatpump * eco_pr.delttime_dayin * sum(dayahead_heatpump) + \
                    eco_pr.cost_watertank * eco_pr.delttime_dayin * sum([abs(x) for x in dayahead_watertank]) + \
                    eco_pr.cost_battery * eco_pr.delttime_dayin * sum([abs(y) for y in dayahead_battery])

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

#
# def penalty_cold_balance(x):
#     # 冷负荷平衡
#     (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
#      dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery,
#      dayahead_gasturbine_heat, dayahead_gasturbine_heat_in) = vari_sepa(x)
#     p_cold_balance = 0
#     for k in range(M):
#         p_cold_balance += (dayahead_absorpchiller[k] + dayahead_elechiller[k] + dayahead_heatpump[k] + dayahead_watertank[k]
#                            - dayahead_cold_load[k]) ** 2
#     return p_cold_balance
#
#
# def penalty_ele_balance(x):
#     # 电负荷平衡
#     (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
#      dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery,
#      dayahead_gasturbine_heat, dayahead_gasturbine_heat_in) = vari_sepa(x)
#     p_ele_balance = 0
#     for k in range(M):
#         p_ele_balance += (dayahead_gasturbine_ele[k] + dayahead_grid[k] + dayahead_battery[k]-
#                           Mag_Elechiller.Magnetic_EleChiller2.get_ele_in(dayahead_elechiller[k])
#                           - HeatPump.VariaFrequency_EleChiller2.get_ele_in(dayahead_heatpump[k]) - dayahead_ele_load[k]) ** 2
#     return p_ele_balance

def penalty_cold_balance(x,k):
    # 冷负荷平衡
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x)

    p_cold_balance = dayahead_cold_load[k] - (dayahead_absorpchiller[k] + dayahead_elechiller[k] + dayahead_heatpump[k] + dayahead_watertank[k])
    return max(0, p_cold_balance-100)


def penalty_ele_balance(x,k):
    # 电负荷平衡
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x)

    p_ele_balance = dayahead_ele_load[k] - (dayahead_gasturbine_ele[k] + dayahead_grid[k] + dayahead_battery[k]
                       - Var_Elechiller.VariaFrequency_EleChiller1.get_ele_in(dayahead_elechiller[k])
                       - HeatPump.heatpump1.get_ele_in(dayahead_heatpump[k]))
    return max(0, p_ele_balance-50)

"""
用vf_elechiller性能曲线模拟heatpump，heatpump模型需要再完善
"""

def demo_func(x):
    m = 10
    n = 10
    l = 10
    p = 0
    q = 0
    r = 0
    for k in range(M):
        p += constraint_absorptionchiller(x, k) + constraint_watertank(x, k) + constraint_battery(x, k)
        q += penalty_cold_balance(x, k)
        r += penalty_ele_balance(x, k)
    # return n*p + l*r
    return cost_min(x) + m*p + n*q +l*r



# 约束
constraint_ueq_list = []

# #gasturbine
# def constraint_gasturbine(x,k):
#     (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
#      dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery,
#      dayahead_gasturbine_heat, dayahead_gasturbine_heat_in) = vari_sepa(x)
#     # constraint_gasturbine = []
#     # gasturbine_ele = dayahead_gasturbine_ele.tolist()  # type:
#         # 爬坡约束
#         # constraint_upper = abs(gasturbine_ele[k]-gasturbine_ele[k-1])-60
#     if k==0:
#         return 0
#     else:
#         constraint_upper = abs(dayahead_gasturbine_ele[k]-dayahead_gasturbine_ele[k-1])-60
#     # constraint_gasturbine.append(constraint_upper)
#         return max(0,constraint_upper)


#absorptionchiller
def constraint_absorptionchiller(x, k):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x)

    dayahead_gasturbine_heat = np.zeros(M, dtype=np.int32)
    dayahead_gasturbine_heat_in = np.zeros(M, dtype=np.int32)
    for u in range(len(dayahead_gasturbine_ele)):
        gt_ele = dayahead_gasturbine_ele[u]
        gt_heat = GT.GasTurbine1.get_heat_in(gt_ele)[1]
        dayahead_gasturbine_heat[u] = gt_heat
        gt_heat_in = GT.GasTurbine1.get_heat_in(gt_ele)[0]
        dayahead_gasturbine_heat_in[u] = gt_heat_in

    #制冷能力上限
    constraint_capacity = Abchiller.AbsorptionChiller2460.get_ele_in(dayahead_absorpchiller[k]) - dayahead_gasturbine_heat[k]
    return max(0, constraint_capacity)

#watertank  蓄能容量上下限
def constraint_watertank(x,k):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x)
    watertank_storage = 0
    for i in range(k):
        #制冷能力上限
        watertank_storage = WaterTank.WaterTank1.get_ColdStorage(dayahead_watertank[i],watertank_storage)

    return max(0, abs(watertank_storage-1000)-1000)
#
#battery 蓄能容量上下限
def constraint_battery(x, k):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x)
    battery_storage = 0
    for i in range(k):
        #制冷能力上限
        battery_storage =  battery.EleStorage1.get_EleStorage(dayahead_battery[i],battery_storage)
    return max(0,abs(battery_storage-1000)-1000)
#         return -watertank_storage,watertank_storage-2000

# watertank_storage = 0
# battery_storage = 0

# for k in range(M):
#     constraint_ueq_list.append(lambda x:constraint_gasturbine(x,k))
#     constraint_ueq_list.append(lambda x:constraint_absorptionchiller(x,k))
#     constraint_ueq_list.append(lambda x:constraint_watertank(x,k))

# pso优化

pso = PSO(func=demo_func, n_dim=n*M, pop=100, max_iter=60, lb=[0,0,0,0,0,-800,-800]*M,w=0.9,c1=2.8,c2=1.2,
          ub=[1200,1200,4920,281.3,5000,0,0]+[1200,1200,4920,281.3,5000,800,800]*(M-1),
                 constraint_ueq=tuple(constraint_ueq_list),verbose=1,T=M)     #[gasturbine,absorpchiller,elechiller,heatpump,grid,watertank,battery] #400→600
        # 假设watertank 释能和蓄能的max power 相同；均为[0,1500]

pso.record_mode = True
pso.run()
print(len(pso.gbest_x))
print(pso.gbest_x)

(dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
 dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(pso.gbest_x)

elechiller_ele_in = np.zeros(M, dtype=np.float32)
heatpump_ele_in = np.zeros(M, dtype=np.float32)
watertank_storage = np.zeros(M, dtype=np.float32)
battery_storage = np.zeros(M, dtype=np.float32)

for k in range(M):
    elechiller_ele_in[k] = Var_Elechiller.VariaFrequency_EleChiller1.get_ele_in(dayahead_elechiller[k])
    heatpump_ele_in[k] = HeatPump.heatpump1.get_ele_in(dayahead_heatpump[k])
    watertank_storage[k] = WaterTank.WaterTank1.get_ColdStorage(dayahead_watertank[k], watertank_storage[k-1])
    battery_storage[k] = battery.EleStorage1.get_EleStorage(dayahead_battery[k], battery_storage[k-1])

print('ele & cold load','\n',
'dayahead_ele_load:',dayahead_ele_load,'\n',
'dayahead_cold_load:',dayahead_cold_load,'\n',
'delechiller_ele_in:',elechiller_ele_in,'\n',
'heatpump_ele_in:',heatpump_ele_in,'\n',
'ele & cold load','\n',
'watertank_storage:',watertank_storage,'\n',
'battery_storage:',battery_storage,'\n',
'power output','\n',
'dayahead_gasturbine_ele:',dayahead_gasturbine_ele,'\n',
'dayahead_absorpchiller',dayahead_absorpchiller,'\n',
'dayahead_elechiller',dayahead_elechiller,'\n',
'dayahead_heatpump',dayahead_heatpump,'\n',
'dayahead_grid',dayahead_grid ,'\n',
'dayahead_watertank',dayahead_watertank,'\n',
'dayahead_battery',dayahead_battery,'\n',
)

# 验证约束
for k in range(M):
    print(constraint_watertank(pso.gbest_x, k))
    print(constraint_battery(pso.gbest_x, k))
    print(constraint_absorptionchiller(pso.gbest_x, k))
    print(penalty_ele_balance(pso.gbest_x, k))
    print(penalty_cold_balance(pso.gbest_x, k))

print(pso.gbest_y)
print(cost_min(pso.gbest_x))
# 作图
# # 画出设备出力曲线
# plt.figure(figsize=(14,6))
# fig, axs = plt.subplots(2, 2, figsize=(18, 12))
# 电负荷情况
plt.subplot(221)
t_list =np.arange(0,M,1)
plt.bar(t_list,dayahead_ele_load,align="center", color="r",label='dayahead_ele_load')
plt.bar(t_list,elechiller_ele_in, align="center", bottom=dayahead_ele_load, color="#66c2a5", label="elechiller_ele_load")
plt.bar(t_list, heatpump_ele_in, align="center", bottom=elechiller_ele_in+dayahead_ele_load, color="#8da0cb", label="heatpump_ele_load")
# plt.plot(t_list, dayahead_gasturbine_ele, color='black',marker='o',label='dayahead_gasturbine_ele')
# plt.plot(t_list,dayahead_grid, color='yellow',marker='o',label='dayahead_grid')
# plt.plot(t_list,dayahead_battery, color='green',marker='o',label='dayahead_battery')
plt.legend(loc='upper left', frameon=False,fontsize=8) # 设置标签
plt.title("ele load", loc='center', fontsize=10, fontweight=0, color='orange') # 设置标题
plt.xlabel("hour",fontsize=8)  # 设置x，y轴
plt.ylabel("ele power",fontsize=8)

plt.subplot(222)
plt.bar(t_list,dayahead_cold_load,label='dayahead_cold_load')
# plt.plot(t_list,dayahead_absorpchiller, color='blue',marker='o',label='dayahead_absorpchiller')
# plt.plot(t_list, dayahead_elechiller, color='green',marker='o',label='dayahead_elechiller')
# plt.plot(t_list,dayahead_heatpump, color='purple',marker='o',label='dayahead_heatpump')
# plt.plot(t_list,dayahead_watertank, color='yellow',marker='o',label='dayahead_watertank')
plt.legend(loc='upper left', frameon=False,fontsize=8) # 设置标签
plt.title("cold load", loc='center', fontsize=10, fontweight=0, color='orange') # 设置标题
plt.xlabel("hour",fontsize=8)  # 设置x，y轴
plt.ylabel("cold power",fontsize=8)




# 电负荷情况
plt.subplot(223)
t_list =np.arange(0,M,1)
width = 0.2
plt.bar(t_list, dayahead_gasturbine_ele, width=width,color='black',label='dayahead_gasturbine_ele')
plt.bar(t_list+width,dayahead_grid, width=width,color='yellow',label='dayahead_grid')
plt.bar(t_list+2*width,dayahead_battery, width=width,color='green',label='dayahead_battery')
plt.legend(loc='lower left', frameon=False,fontsize=6) # 设置标签
plt.subplot(224)
plt.bar(t_list,dayahead_absorpchiller, width=width,color='blue',label='dayahead_absorpchiller')
plt.bar(t_list+width, dayahead_elechiller, width=width,color='green',label='dayahead_elechiller')
plt.bar(t_list+2*width,dayahead_heatpump, width=width,color='purple',label='dayahead_heatpump')
plt.bar(t_list+3*width,dayahead_watertank,width=width, color='yellow',label='dayahead_watertank')
plt.legend(loc='lower left', frameon=False,fontsize=6) # 设置标签

plt.tight_layout()
plt.show()