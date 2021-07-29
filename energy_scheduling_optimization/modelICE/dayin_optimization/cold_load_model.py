import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank

# 负荷
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

# 日内负荷预测
np.random.seed(0)
dayin_load = np.zeros(24*4,dtype=np.int32)


for t in range(24):
    mean = dayahead_load[t]
    standard_deviation = mean * 0.1

    for i in range(4):
        cur = standard_deviation * np.random.randn() + mean
        dayin_load[4*t+i] = cur
print(dayin_load)
print(len(dayin_load))



# 绘制日前&日内预测曲线
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

y1 = list(dayahead_load)
y2 = list(dayin_load)
x1 = [i for i in range(24)]
#x2 = list(np.arange(0.0, 24.0, 0.25))
x2 = list(np.arange(0.0, 23.0, 0.25))
plt.step(x1, y1, color='green', label='day ahead load')
plt.step(x2, y2, color='red', label='day in load')

# 修改密度 设置间隔
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.show()



def fluc_min(x,t):
    dayin_CHP_t_i = x[0]
    dayin_heatpump_t_i = x[1]
    dayin_chiller_t_i = x[2]
    dayin_coldtank_i = x[3]   #watertank 为正 表示释能； 为负 表示蓄能
    fluc_squa = (dayahead_CHP[t] - dayin_CHP_t_i) ** 2 + \
                (dayahead_heatpump[t] - dayin_heatpump_t_i) ** 2 + \
                (dayahead_chiller[t] - dayin_chiller_t_i) ** 2 + \
                (dayahead_watertank[t] - dayin_coldtank_i) ** 2
    return fluc_squa

def penalty_cold_balance(x,t,i):
    p_cold_balance = abs(x[0]+x[1]+x[2]+x[3]-dayin_load[4*t+i])
    return p_cold_balance

def demo_func(x):
    m = 100000
    return fluc_min(x,t) + m * penalty_cold_balance(x,t,i)

# !!!!

constraint_ueq_list = []

#  watertank 容量上下限约束
constraint_ueq_list.append(lambda x: WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0)-1500)   #容量上限
constraint_ueq_list.append(lambda x: -WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0))   #容量下限

# watertank 时间终点容量 = 时间起点容量 约束
#？？？ 现在是逐时优化，如果要确保等式成立，需要将所有时间设备出力作为一维变量同时优化？

# CHP 爬坡约束
#def constraint_CHP(x):
# 功率是指发电量吗？

# heatpump 设备启停约束
constraint_ueq_list.append(lambda x: x[1]-0.1 if x[1] < 200 else x[1]-500)  # 下限在lb已体现 无需约束
# 可参考demo3
# def constraint_heatpump(x):
#     global heatpump_cons1, heatpump_cons2
#     if x[1] < 200:
#         heatpump_cons1 = x[1] - 0.1
#         heatpump_cons2 = -x[1]
#     return heatpump_cons1,heatpump_cons2

best_x_list = []
best_y_list = []
dayin_load_t = []
dayin_supply_t = []



coldstorage_t0 = 0
coldstorage_start = coldstorage_t0
dayin_CHP = np.zeros(24*4,dtype=np.int32)
dayin_heatpump = np.zeros(24*4,dtype=np.int32)
dayin_chiller = np.zeros(24*4,dtype=np.int32)
dayin_watertank = np.zeros(24*4,dtype=np.int32)
dayin_watertank_storage = np.zeros(24*4,dtype=np.int32)

#for t in range(24):
t=10
for i in range(4):
    #global best_x_list,best_y_list
    pso = PSO(func=demo_func, n_dim=4, pop=1000, max_iter=1000, lb=[0,0,0,-1500], ub=[1500,500,8500,1500],
              constraint_ueq=tuple(constraint_ueq_list), verbose=1)     #[CHP,heatpump,chiller,watertank]
    # 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
    pso.record_mode = True
    pso.run()
    # 储存下各时刻设备的出力情况 & watertank_storage
    dayin_CHP[4*t+i] = pso.gbest_x[0]
    dayin_heatpump[4*t+i] = pso.gbest_x[1]
    dayin_chiller[4*t+i] = pso.gbest_x[2]
    dayin_watertank[4*t+i] = pso.gbest_x[3]   #watertank 为正 表示释能； 为负 表示蓄能
    coldstorage_t0 = WaterTank.ColdStorage1.get_ColdStorage(dayin_watertank[4*t+i],coldstorage_t0)
    dayin_watertank_storage[4*t+i] = coldstorage_t0

    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    print(fluc_min(pso.gbest_x,t),penalty_cold_balance(pso.gbest_x,t,i))
    dayin_load_t.append(dayin_load[4 * t + i])
    dayin_supply_t.append(sum(pso.gbest_x[i] for i in range(3)))
    best_x_list.append(list(pso.gbest_x))
    best_y_list.append(list(pso.gbest_y))

print('10h_dayahead_power:',dayahead_CHP[10],dayahead_heatpump[10],dayahead_chiller[10],dayahead_watertank[10])  #!!
print('10h_dayin_power:',best_x_list)
print('demo_func_min:',best_y_list)
print('dayin_load_t:',dayin_load_t)
print('dayin_supply_t:',dayin_supply_t)


# print(fluc_min(pso.gbest_x))
# print(penalty_cold_balance(pso.gbest_x))
# print(dayahead_CHP[10],dayahead_heatpump[10],dayahead_chiller[10])
# print(dayahead_CHP[10]-pso.gbest_x[0],dayahead_heatpump[10]-pso.gbest_x[1],dayahead_chiller[10]-pso.gbest_x[2])

# 设备日内出力曲线