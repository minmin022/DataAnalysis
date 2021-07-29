import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 负荷
dayahead_ele_load_list = [822, 798, 803, 809, 851, 971, 1103, 1211, 1282, 1306, 1306,1300,
     1287, 1281, 1263, 1245, 1214, 1214, 1130, 1064, 1009, 931, 901, 852]
dayahead_cold_load_list = [722, 711, 699, 676, 676, 674, 678, 762, 949, 976, 1081, 1162,
     1265, 1331, 1362, 1367, 1310, 1309, 986, 926, 869, 766, 748, 732]  # 空间冷负荷 + 500kw冷冻负荷
dayahead_ele_load = np.array(dayahead_ele_load_list)
dayahead_cold_load = np.array(dayahead_cold_load_list)
print("day ahead ele load:",dayahead_ele_load)
print("day ahead cold load:",dayahead_cold_load)

# 设备日前出力调度计划
# 燃气轮机
dayahead_gasturbine_list = [161, 143, 143, 154, 196, 412, 634, 718, 850, 910, 898, 1041,
     1065, 1077, 1071, 1022, 908, 824, 800, 787, 757, 715, 246, 186]
dayahead_gasturbine = np.array(dayahead_gasturbine_list)
print("day ahead gasturbine power:",dayahead_gasturbine)

# 热泵 & 电制冷
dayahead_elecold_list = [552, 577, 601, 613, 614, 602, 488, 536, 482, 447, 399, 369,
     321, 291, 280, 292, 316, 352, 221, 317, 329, 281, 522, 546]
dayahead_elecold_load = np.array(dayahead_elecold_list)
dayahead_heatpump_list = [46, 42, 42, 42, 43, 49, 55, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 48]
dayahead_heatpump = np.array(dayahead_heatpump_list)
print("day ahead heatpump power:",dayahead_heatpump)
dayahead_elechiller = dayahead_elecold_load - dayahead_heatpump
print("day ahead elechiller power:",dayahead_elechiller)
#电网
dayahead_grid = dayahead_ele_load + dayahead_elecold_load- dayahead_gasturbine
print("day ahead grid power:",dayahead_grid)
#吸收式制冷
dayahead_absorpchiller = dayahead_cold_load - dayahead_elecold_load
print("day ahead absorpchiller power:",dayahead_absorpchiller)

# 日内负荷预测
np.random.seed(0)
dayin_cold_load = np.zeros(24*4,dtype=np.int32)
for t in range(24):
    mean = dayahead_cold_load[t]
    standard_deviation = mean * 0.1

    for i in range(4):
        cur = standard_deviation * np.random.randn() + mean
        dayin_cold_load[4*t+i] = cur
print(dayin_cold_load)
print(len(dayin_cold_load))

np.random.seed(1)
dayin_ele_load = np.zeros(24*4,dtype=np.int32)
for t in range(24):
    mean = dayahead_ele_load[t]
    standard_deviation = mean * 0.1

    for i in range(4):
        cur = standard_deviation * np.random.randn() + mean
        dayin_ele_load[4*t+i] = cur
print(dayin_ele_load)
print(len(dayin_ele_load))


# #绘制日前&日内预测曲线
# # import matplotlib.pyplot as plt
# # from matplotlib.pyplot import MultipleLocator
#
# y1 = list(dayahead_ele_load)
# y2 = list(dayin_ele_load)
# x1 = [i for i in range(24)]
# #x2 = list(np.arange(0.0, 24.0, 0.25))
# x2 = list(np.arange(0.0, 24.0, 0.25))
# plt.step(x1, y1, color='green', label='day ahead ele load')
# plt.step(x2, y2, color='red', label='day in ele load')
# plt.figure()
# #plt.subplot(211)
#
#
#
# y3 = list(dayahead_cold_load)
# y4 = list(dayin_cold_load)
# x1 = [i for i in range(24)]
# #x2 = list(np.arange(0.0, 24.0, 0.25))
# x2 = list(np.arange(0.0, 24.0, 0.25))
# plt.step(x1, y3, color='blue', label='day ahead cold load')
# plt.step(x2, y4, color='yellow', label='day in cold load')
# plt.figure()
# #plt.subplot(212)
# # 修改密度 设置间隔
# x_major_locator=MultipleLocator(1)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
#
# plt.show()





def fluc_min(x,t):
    dayin_gasturbine_t_i = x[0]
    dayin_absorpchiller_t_i = x[1]
    dayin_elechiller_t_i = x[2]
    dayin_heatpump_t_i = x[3]
    dayin_grid_t_i = x[4]
    #dayin_coldtank_i = x[3]   #watertank 为正 表示释能； 为负 表示蓄能
    fluc_squa = (dayahead_gasturbine[t] - dayin_gasturbine_t_i) ** 2 + \
                (dayahead_absorpchiller[t] - dayin_absorpchiller_t_i) ** 2 + \
                (dayahead_elechiller[t] - dayin_elechiller_t_i) ** 2 + \
                (dayahead_heatpump[t] - dayin_heatpump_t_i) ** 2 + \
                (dayahead_grid[t] - dayin_grid_t_i) ** 2
    return fluc_squa

def penalty_demand_balance(x,t,i):
    p_cold_balance = abs(x[1]+x[2]+x[3]-dayin_cold_load[4*t+i])
    p_ele_balance = abs(x[0]+x[4]-dayin_ele_load[4*t+i])
    return p_cold_balance + p_ele_balance

def demo_func(x):
    m = 1000
    return fluc_min(x,t) + m * penalty_demand_balance(x,t,i)

# !!!!

# constraint_ueq_list = []

# #  watertank 容量上下限约束
# constraint_ueq_list.append(lambda x: WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0)-1500)   #容量上限
# constraint_ueq_list.append(lambda x: -WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0))   #容量下限

# watertank 时间终点容量 = 时间起点容量 约束
#？？？ 现在是逐时优化，如果要确保等式成立，需要将所有时间设备出力作为一维变量同时优化？

# CHP 爬坡约束
#def constraint_CHP(x):
# 功率是指发电量吗？


# 日前才需要考虑启停约束
# # heatpump 设备启停约束
# constraint_ueq_list.append(lambda x: x[3]-0.1 if x[3] < 200 else x[3]-500)  # 下限在lb已体现 无需约束
# 可参考demo3
# def constraint_heatpump(x):
#     global heatpump_cons1, heatpump_cons2
#     if x[1] < 200:
#         heatpump_cons1 = x[1] - 0.1
#         heatpump_cons2 = -x[1]
#     return heatpump_cons1,heatpump_cons2

best_x_list = []
best_y_list = []
dayin_ele_load_t = []
dayin_cold_load_t = []
dayin_ele_supply_t = []
dayin_cold_supply_t = []



# coldstorage_t0 = 0
# coldstorage_start = coldstorage_t0
dayin_gasturbine = np.zeros(24*4,dtype=np.int32)
dayin_absorpchiller = np.zeros(24*4,dtype=np.int32)
dayin_elechiller = np.zeros(24*4,dtype=np.int32)
dayin_heatpump = np.zeros(24*4,dtype=np.int32)
dayin_grid = np.zeros(24*4,dtype=np.int32)

# dayin_watertank = np.zeros(24*4,dtype=np.int32)
# dayin_watertank_storage = np.zeros(24*4,dtype=np.int32)

#for t in range(24):
t=10
for i in range(4):
    #global best_x_list,best_y_list
    constraint_ueq_list = []
    # 启停约束
    constraint_ueq_list.append(lambda x, t=t: x[0] - 0.1 if dayahead_gasturbine[t] == 0 else x[0] - 1500)
    constraint_ueq_list.append(lambda x, t=t: x[1] - 0.1 if dayahead_absorpchiller[t] == 0 else x[1] - 1500)
    constraint_ueq_list.append(lambda x, t=t: x[2] - 0.1 if dayahead_elechiller[t] == 0 else x[2] - 800)
    constraint_ueq_list.append(lambda x, t=t: x[3] - 0.1 if dayahead_heatpump[t] == 0 else x[3] - 500)
    constraint_ueq_list.append(lambda x, t=t: x[4] - 0.1 if dayahead_grid[t] == 0 else x[4] - 1500)
    # 爬坡约束
    # 偏差约束
    constraint_ueq_list.append(lambda x, t=t: abs(x[0] - dayahead_gasturbine[t]) - 200)
    constraint_ueq_list.append(lambda x, t=t: abs(x[1] - dayahead_absorpchiller[t]) - 200)
    constraint_ueq_list.append(lambda x, t=t: abs(x[2] - dayahead_elechiller[t]) - 200)
    constraint_ueq_list.append(lambda x, t=t: abs(x[3] - dayahead_heatpump[t]) - 200)
    constraint_ueq_list.append(lambda x, t=t: abs(x[4] - dayahead_grid[t]) - 200)

    pso = PSO(func=demo_func, n_dim=5, pop=2000, max_iter=1000, lb=[0,0,0,0,0], ub=[1500,1500,800,500,2000],
              constraint_ueq=tuple(constraint_ueq_list), verbose=1)     #[gasturbine,absorpchiller,elechiller,heatpump,grid]
    # 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
    pso.record_mode = True
    pso.run()
    # 储存下各时刻设备的出力情况 & watertank_storage
    dayin_gasturbine[4*t+i] = pso.gbest_x[0]
    dayin_absorpchiller[4*t+i] = pso.gbest_x[1]
    dayin_elechiller[4*t+i] = pso.gbest_x[2]
    dayin_heatpump[4 * t + i] = pso.gbest_x[3]
    dayin_grid[4 * t + i] = pso.gbest_x[4]
    # dayin_watertank[4*t+i] = pso.gbest_x[3]   #watertank 为正 表示释能； 为负 表示蓄能
    # coldstorage_t0 = WaterTank.ColdStorage1.get_ColdStorage(dayin_watertank[4*t+i],coldstorage_t0)
    # dayin_watertank_storage[4*t+i] = coldstorage_t0

    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    print(fluc_min(pso.gbest_x,t), penalty_demand_balance(pso.gbest_x,t,i))
    dayin_ele_load_t.append(dayin_ele_load[4 * t + i])
    dayin_cold_load_t.append(dayin_cold_load[4 * t + i])
    dayin_ele_supply_t.append(sum(pso.gbest_x[i] for i in (0,4)))
    dayin_cold_supply_t.append(sum(pso.gbest_x[i] for i in (1,2,3)))
    best_x_list.append(list(pso.gbest_x))
    best_y_list.append(list(pso.gbest_y))

print('10h_dayahead_ele_load:',dayahead_ele_load[10])
print('10h_dayahead_cold_load:',dayahead_cold_load[10])
print('10h_dayahead_power:',dayahead_gasturbine[10],dayahead_absorpchiller[10],
      dayahead_elechiller[10],dayahead_heatpump[10],dayahead_grid[10])  #!!
# print('10h_dayahead_power:',dayahead_gasturbine[10],dayahead_heatpump[10],dayahead_chiller[10],dayahead_watertank[10])  #!!
print('10h_dayin_power:',best_x_list)
print('demo_func_min:',best_y_list)
print('dayin_ele_load_t:',dayin_ele_load_t)
print('dayin_ele_supply_t:',dayin_ele_supply_t)
print('dayin_cold_load_t:',dayin_cold_load_t)
print('dayin_cold_supply_t:',dayin_cold_supply_t)


# print(fluc_min(pso.gbest_x))
# print(penalty_cold_balance(pso.gbest_x))
# print(dayahead_gasturbine[10],dayahead_heatpump[10],dayahead_chiller[10])
# print(dayahead_gasturbine[10]-pso.gbest_x[0],dayahead_heatpump[10]-pso.gbest_x[1],dayahead_chiller[10]-pso.gbest_x[2])

# 设备日内出力曲线