"""
自适应控制步长
1h为控制步长，M为预测时域 假设为1h

目标方程 仅为下一小时的运行成本 min
优化24次 得到24h总成本 min
变量数目固定，为未来24h*n个设备变量
自适应时域 用于满足 约束
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


M = 4
# dayin_CHP = np.zeros(M*4,dtype = np.int32)
# dayin_heatpump= np.zeros(M*4,dtype = np.int32)
# dayin_chiller = np.zeros(M*4,dtype = np.int32)
# dayin_coldtank = np.zeros(M*4,dtype = np.int32)

# cost_min = 0
def cost_min(x):  #x为4*M列表
# def demo_func(x):
    # global dayin_CHP,dayin_heatpump,dayin_chiller,dayin_coldtank,cost_min
    # cost_min = 0
    dayin_CHP = []
    dayin_heatpump = []
    dayin_chiller = []
    dayin_coldtank = []
    for m in range(len(x)-1):
        if m % 4 == 0:
            dayin_CHP.append(x[m])    # M
            # dayin_CHP[]
        elif m % 4 == 1:
            dayin_heatpump.append(x[m])
        elif m % 4 == 2:
            dayin_chiller.append(x[m])
        else:
            dayin_coldtank.append(x[m])   #watertank 为正 表示释能； 为负 表示蓄能

    # for k in range(4*M):
    # dayin_CHP_0 = dayin_CHP[0]
    # dayin_heatpump_0 = dayin_heatpump[0]
    # dayin_chiller_0 = dayin_chiller[0]
    # dayin_coldtank_0 = dayin_coldtank[0]

    # cost_min = 3000 * dayin_CHP_0 + 970 * dayin_heatpump_0 + 1200 * dayin_chiller_0 + 230 * abs(dayin_coldtank_0)
    cost_min = (3000 * sum(dayin_CHP) + 970 * sum(dayin_heatpump) + 1200 * sum(dayin_chiller) + 230 * sum(dayin_coldtank))/M
        # cost_CHP
        # cost_heatpump
        # cost_chiller
        # cost_coldtank
    return cost_min

def penalty_cold_balance(x,t):
    # dayin_load
    # 日内负荷预测
    M = int(x[-1])
    global dayin_load
    np.random.seed(0)
    dayin_load = np.zeros(M, dtype=np.int32)

    for k in range(M):
        mean = dayahead_load[t+M]
        standard_deviation = mean * 0.1

        # for i in range(4):
        cur = standard_deviation * np.random.randn() + mean
        dayin_load[k] = cur
    # print(dayin_load)
    # print(len(dayin_load))

    # 冷负荷平衡
    p_cold_balance = 0
    for k in range(M):
        dayin_power_k = x[4*k:4*k+4]
        p_cold_balance += (sum(dayin_power_k)- dayin_load[k]) ** 2
        # p_cold_balance += abs(dayin_CHP[i]+dayin_heatpump[i]+dayin_chiller[i]+dayin_coldtank[i]-dayin_load[i])

    return p_cold_balance

def demo_func(x):
    m = 100000
    return cost_min(x) + m * penalty_cold_balance(x,t)


ub_list = [3000, 500, 8500, 1500] * M + [8]
lb_list = [0, 0, 0, -1500] * M + [1]
t = 10
pso = PSO(func=demo_func, n_dim=4*M+1, pop=1000, max_iter=500, lb=lb_list, ub=ub_list,
           verbose=1)  # [CHP,heatpump,chiller,watertank]
# 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
pso.record_mode = True
pso.run()

dayin_power_list = []
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
for k in range(8):
    dayin_power_k = pso.gbest_x[4 * k:4 * k + 4]
    dayin_power_list.append(sum(dayin_power_k))
print("day ahead cold load:",dayahead_load[t:t+8])
print('dayin_power_list:',dayin_power_list)
print('dayin_load:',dayin_load)




