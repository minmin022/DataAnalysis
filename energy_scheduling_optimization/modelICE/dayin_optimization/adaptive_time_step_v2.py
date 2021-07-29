"""
自适应控制步长
15min为控制步长，1h为控制时域，M为预测时域 假设为1h
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


M = 1
# dayin_CHP = np.zeros(M*4,dtype = np.int32)
# dayin_heatpump= np.zeros(M*4,dtype = np.int32)
# dayin_chiller = np.zeros(M*4,dtype = np.int32)
# dayin_coldtank = np.zeros(M*4,dtype = np.int32)

# cost_min = 0
def cost_min(x):  #x为4*4*4列表
# def demo_func(x):
    # global dayin_CHP,dayin_heatpump,dayin_chiller,dayin_coldtank,cost_min
    cost_min = 0
    dayin_CHP = []
    dayin_heatpump = []
    dayin_chiller = []
    dayin_coldtank = []
    for m in range(len(x)):
        if m % 4 == 0:
            dayin_CHP.append(x[m])    # 4*M
            # dayin_CHP[]
        elif m % 4 == 1:
            dayin_heatpump.append(x[m])
        elif m % 4 == 2:
            dayin_chiller.append(x[m])
        else:
            dayin_coldtank.append(x[m])   #watertank 为正 表示释能； 为负 表示蓄能

    # for k in range(4*M):
    #     dayin_CHP_k = dayin_CHP[k]
    #     dayin_heatpump_k = dayin_heatpump[k]
    #     dayin_chiller_k = dayin_chiller[k]
    #     dayin_coldtank_k = dayin_coldtank[k]
    #
    #     cost_min += 3000 * dayin_CHP_k + 500 * dayin_heatpump_k + 800 * dayin_chiller_k + 1500 * dayin_coldtank_k
    cost_min = 3000 * sum(dayin_CHP) + 500 * sum(dayin_heatpump) + 800 * sum(dayin_chiller) + 1500 * sum(dayin_coldtank)
        # cost_CHP
        # cost_heatpump
        # cost_chiller
        # cost_coldtank
    return cost_min

def penalty_cold_balance(x,t):
    # dayin_load
    # 日内负荷预测
    global dayin_load
    np.random.seed(0)
    dayin_load = np.zeros(M * 4, dtype=np.int32)

    for k in range(M):
        mean = dayahead_load[t+M]
        standard_deviation = mean * 0.1

        for i in range(4):
            cur = standard_deviation * np.random.randn() + mean
            dayin_load[4 * k + i] = cur
    # print(dayin_load)
    # print(len(dayin_load))

    # 冷负荷平衡
    p_cold_balance = 0
    for k in range(4*M):
        dayin_power_k = x[4*k:4*k+4]
        p_cold_balance += (sum(dayin_power_k)- dayin_load[k])**2
        # p_cold_balance += abs(dayin_CHP[i]+dayin_heatpump[i]+dayin_chiller[i]+dayin_coldtank[i]-dayin_load[i])

    return p_cold_balance

def demo_func(x):
    m = 100000
    return cost_min(x) + m * penalty_cold_balance(x,t)

lb_list = [0, 0, 0, 0] * M * 4
ub_list = [1500, 500, 8500, 1500] * M * 4
t = 10
pso = PSO(func=demo_func, n_dim=4*4*M, pop=1000, max_iter=1000, lb=lb_list, ub=ub_list,
           verbose=1)  # [CHP,heatpump,chiller,watertank]
# 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
pso.record_mode = True
pso.run()

dayin_power_list = []
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
for k in range(4 * M):
    dayin_power_k = pso.gbest_x[4 * k:4 * k + 4]
    dayin_power_list.append(sum(dayin_power_k))
print(dayin_power_list)
print(dayin_load)




