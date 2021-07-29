"""
自适应控制步长

输入：电冷负荷波动频率&幅度
输出：短时域控制时间步长
算法：梯度下降算法
目标方程：
定义代价函数：均方误差代价函数
google：	Self-adaptive step
知网：步长自调整
"""
import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank

# 日前负荷
dayahead_load_list = [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 8600, 8800,9000,
     9100, 9000, 9100, 9200, 9300, 9500, 9450, 9300, 8400, 8300, 1200, 1200]
dayahead_load = np.array(dayahead_load_list)
print("day ahead cold load:",dayahead_load)

# 设备日前出力调度计划
dayahead_CHP=1200 * np.ones(24,dtype=np.int32)
for i in range(7):
    dayahead_CHP[i] = 2400
print("day ahead CHP power:",dayahead_CHP)

dayahead_heatpump = np.zeros(24,dtype = np.int32)
for i in range(24):
    if 9 <= i <= 21:
        dayahead_heatpump[i] = 200
print("day ahead heatpump power:",dayahead_heatpump)

dayahead_chiller = np.zeros(24,dtype = np.int32)
for i in range(24):
    if 9 <= i <= 21:
        dayahead_chiller[i] = 6800
print("day ahead chiller power:",dayahead_chiller)

dayahead_watertank = dayahead_load - (dayahead_CHP + dayahead_heatpump + dayahead_chiller)  #watertank 为正 表示释能；为负 表示蓄能
print("day ahead watertank power:",dayahead_watertank)


#目标方程 预测时域M的运行成本最小
# CHP heatpump chiller watertank
# 先假设预测时域 M = 4
dayin_CHP = []
dayin_heatpump = []
dayin_chiller = []
dayin_coldtank = []
M = 1
t = 10


# dayin_load
# 日内负荷预测   #有没有必要把预测放在目标函数里
np.random.seed(0)
dayin_load = np.zeros(M * 4, dtype=np.int32)

for k in range(M):
    mean = dayahead_load[t + k]
    print(mean)
    standard_deviation = mean * 0.1

    for i in range(4):
        cur = standard_deviation * np.random.randn() + mean
        dayin_load[4 * k + i] = cur
print(dayin_load)
print(len(dayin_load))

cost_min_v = 0
def cost_min(x):  #x为4*4*M列表
# def demo_func(x):
    global dayin_CHP,dayin_heatpump,dayin_chiller,dayin_coldtank,cost_min_v
    for m in range(4*4*M):
        if m % 4 == 0:
            dayin_CHP.append(x[m])    # 4*M
        if m % 4 == 1:
            dayin_heatpump.append(x[m])
        if m % 4 == 2:
            dayin_chiller.append(x[m])
        else:
            dayin_coldtank.append(x[m])   #watertank 为正 表示释能； 为负 表示蓄能
    for k in range(4*M):
        dayin_CHP_k = dayin_CHP[k]
        dayin_heatpump_k = dayin_heatpump[k]
        dayin_chiller_k = dayin_chiller[k]
        dayin_coldtank_k = dayin_coldtank[k]

        cost_min_v += 3000 * dayin_CHP_k + 500 * dayin_heatpump_k + 800 * dayin_chiller_k + 1500 * dayin_coldtank_k
        # cost_CHP
        # cost_heatpump
        # cost_chiller
        # cost_coldtank
    return cost_min_v

def penalty_cold_balance():
    # # dayin_load
    # # 日内负荷预测   #有没有必要把预测放在目标函数里
    # np.random.seed(0)
    # dayin_load = np.zeros(M * 4, dtype=np.int32)
    #
    # for k in range(M):
    #     mean = dayahead_load[t+M]
    #     standard_deviation = mean * 0.1
    #
    #     for i in range(4):
    #         cur = standard_deviation * np.random.randn() + mean
    #         dayin_load[4 * k + i] = cur
    # print(dayin_load)
    # print(len(dayin_load))

    # 冷负荷平衡
    p_cold_balance_i0 = (dayin_CHP[0]+dayin_heatpump[0]+dayin_chiller[0]+dayin_coldtank[0]-dayin_load[0])**2
    p_cold_balance_i1 = (dayin_CHP[1] + dayin_heatpump[1] + dayin_chiller[1] + dayin_coldtank[1] - dayin_load[1]) ** 2
    p_cold_balance_i2 = (dayin_CHP[2] + dayin_heatpump[2] + dayin_chiller[2] + dayin_coldtank[2] - dayin_load[2]) ** 2
    p_cold_balance_i3 = (dayin_CHP[3] + dayin_heatpump[3] + dayin_chiller[3] + dayin_coldtank[3] - dayin_load[3]) ** 2
    p_cold_balance = p_cold_balance_i0 + p_cold_balance_i1 + p_cold_balance_i2 + p_cold_balance_i3
    return p_cold_balance

def demo_func(x):
    m = 100000
    return cost_min(x) + m * penalty_cold_balance()
#
# constraint_ueq_list = []
# # 偏差约束
# # for i in range(4):
#     # power_out[i] = dayin_CHP[i]+dayin_heatpump[i]+dayin_chiller[i]+dayin_coldtank[i]
# constraint_ueq_list.append(lambda x, t=t: abs(x[0]+x[1]+x[2]+x[3]- dayin_load[0]-1))
# constraint_ueq_list.append(lambda x, t=t: abs(x[4]+x[5]+x[6]+x[7]- dayin_load[1]-1))
# constraint_ueq_list.append(lambda x, t=t: abs(x[8]+x[9]+x[10]+x[11]- dayin_load[2]-1))
# constraint_ueq_list.append(lambda x, t=t: abs(x[12]+x[13]+x[14]+x[15]- dayin_load[3]-1))
#
# constraint_ueq_list.append(lambda x, t=t: abs(-(x[0]+x[1]+x[2]+x[3]) + dayin_load[0]))
# constraint_ueq_list.append(lambda x, t=t: abs(-(x[4]+x[5]+x[6]+x[7]) + dayin_load[1]))
# constraint_ueq_list.append(lambda x, t=t: abs(-(x[8]+x[9]+x[10]+x[11]) + dayin_load[2]))
# constraint_ueq_list.append(lambda x, t=t: abs(-(x[12]+x[13]+x[14]+x[15]) + dayin_load[3]))
# # constraint_ueq_list.append(lambda x, t=t: abs(x[4] - dayahead_grid[t]) - 200)


lb_list = [0, 0, 0, 0] * M * 4
ub_list = [1500, 500, 8500, 1500] * M * 4
t = 10
pso = PSO(func=demo_func, n_dim=4*4*M, pop=1000, max_iter=500,
          lb=lb_list, ub=ub_list,
           # constraint_ueq=tuple(constraint_ueq_list),
          verbose=1)  # [CHP,heatpump,chiller,watertank]
# 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
pso.record_mode = True
pso.run()

# 调通了 但是不迭代寻优


