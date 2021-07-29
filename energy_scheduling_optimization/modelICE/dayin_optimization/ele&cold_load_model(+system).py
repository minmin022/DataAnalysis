import numpy as np
from modelICE.PSO import PSO
from modelICE.PSO.PSO_modify import PSO
from modelICE.model import WaterTank
from modelICE.model import GasTurbine as GT
from modelICE.model import Magnetic_EleChiller as Mag_Elechiller
from modelICE.model import AbsorptionChiller as Abchiller
from modelICE.model import VariaFrequency_EleChiller as HeatPump
from modelICE.Parameters import economy_parameters as eco_pr
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 负荷
# dayahead_ele_load_list = [822, 798, 803, 809, 851, 971, 1103, 1211, 1282, 1306, 1306,1300,
#      1287, 1281, 1263, 1245, 1214, 1214, 1130, 1064, 1009, 931, 901, 852]
# dayahead_cold_load_list = [722, 711, 699, 676, 676, 674, 678, 762, 949, 976, 1081, 1162,
#      1265, 1331, 1362, 1367, 1310, 1309, 986, 926, 869, 766, 748, 732]  # 空间冷负荷 + 500kw冷冻负荷
dayahead_ele_load_list = [822, 798, 803, 809, 851, 971, 1103, 1211, 1282, 1306, 1306,1300,
     1287, 1281, 1263, 1245, 1214, 1214, 1130, 1064, 1009, 931, 901, 852,822, 798, 803,809,851,971, 1103]
dayahead_cold_load_list = [722, 711, 699, 676, 676, 674, 678, 762, 949, 976, 1081, 1162,
     1265, 1331, 1362, 1367, 1310, 1309, 986, 926, 869, 766, 748, 732,722, 711, 699,676,676,674, 678]  # 空间冷负荷 + 500kw冷冻负荷
dayahead_ele_load = np.array(dayahead_ele_load_list)
dayahead_cold_load = np.array(dayahead_cold_load_list)
print("day ahead ele load:",dayahead_ele_load)
print("day ahead cold load:",dayahead_cold_load)



# 设备日前出力调度计划
# 燃气轮机
# dayahead_gasturbine_list = [161, 143, 143, 154, 196, 412, 634, 718, 850, 910, 898, 1041,
#      1065, 1077, 1071, 1022, 908, 824, 800, 787, 757, 715, 246, 186]
dayahead_gasturbine_list = [161, 143, 143, 154, 196, 412, 634, 718, 850, 910, 898, 1041,
     1065, 1077, 1071, 1022, 908, 824, 800, 787, 757, 715, 246, 186,161, 143, 143,154,196,412, 634]
dayahead_gasturbine = np.array(dayahead_gasturbine_list)
print("day ahead gasturbine power:",dayahead_gasturbine)

# 热泵 & 电制冷
# dayahead_elecold_load_list = [552, 577, 601, 613, 614, 602, 488, 536, 482, 447, 399, 369,
#      321, 291, 280, 292, 316, 352, 221, 317, 329, 281, 522, 546]
dayahead_elecold_load_list = [552, 577, 601, 613, 614, 602, 488, 536, 482, 447, 399, 369,
     321, 291, 280, 292, 316, 352, 221, 317, 329, 281, 522, 546,552, 577, 601,613,614,602, 488]
dayahead_elecold_load = np.array(dayahead_elecold_load_list)
# dayahead_heatpump_list = [46, 42, 42, 42, 43, 49, 55, 0, 0, 0, 0, 0,
#      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 48]
dayahead_heatpump_list = [46, 42, 42, 42, 43, 49, 55, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 48,46, 42, 42,42,43, 49, 55]
dayahead_heatpump = np.array(dayahead_heatpump_list)
print("day ahead heatpump power:",dayahead_heatpump)
dayahead_elechiller = dayahead_elecold_load - dayahead_heatpump
print("day ahead elechiller power:",dayahead_elechiller)
#电网
dayahead_grid = dayahead_ele_load + dayahead_elecold_load - dayahead_gasturbine
print("day ahead grid power:",dayahead_grid)
#吸收式制冷
dayahead_absorpchiller = dayahead_cold_load - dayahead_elecold_load
print("day ahead absorpchiller power:",dayahead_absorpchiller)

# # 日内负荷预测
# np.random.seed(0)
# dayin_cold_load = np.zeros(24*4,dtype=np.int32)
# for t in range(24):
#     mean = dayahead_cold_load[t]
#     standard_deviation = mean * 0.1
#
#     for i in range(4):
#         cur = standard_deviation * np.random.randn() + mean
#         dayin_cold_load[4*t+i] = cur
# print(dayin_cold_load)
# print(len(dayin_cold_load))
#
# np.random.seed(1)
# dayin_ele_load = np.zeros(24*4,dtype=np.int32)
# for t in range(24):
#     mean = dayahead_ele_load[t]
#     standard_deviation = mean * 0.1
#
#     for i in range(4):
#         cur = standard_deviation * np.random.randn() + mean
#         dayin_ele_load[4*t+i] = cur
# print(dayin_ele_load)
# print(len(dayin_ele_load))
#
#
# #绘制日前&日内预测曲线
# # import matplotlib.pyplot as plt
# # from matplotlib.pyplot import MultipleLocator
#
# y1 = list(dayahead_ele_load)
# # y2 = list(dayin_ele_load)
# x1 = [i for i in range(24)]
# x2 = list(np.arange(0.0, 24.0, 0.25))
# x2 = list(np.arange(0.0, 24.0, 0.25))
# plt.step(x1, y1, color='green', label='day ahead ele load')
# plt.step(x2, y2, color='red', label='day in ele load')
# plt.figure()
# #plt.subplot(211)
# #
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


# M = 7 # 预测时域长度为4
total_cost_m_t = np.zeros(24, dtype=np.int32)
total_gasturbine_m_t = np.zeros(24, dtype=np.int32)
total_absorpchiller_m_t = np.zeros(24, dtype=np.int32)
total_elechiller_m_t = np.zeros(24, dtype=np.int32)
total_heatpump_m_t = np.zeros(24, dtype=np.int32)
total_grid_m_t = np.zeros(24, dtype=np.int32)
n = 5  #5个设备
# M = 6
for M in (1,2):
    for t in range(0,24,1):

        # 日内负荷预测
        #电负荷
        np.random.seed(0)
        dayin_ele_load = np.zeros(M, dtype=np.int32)
        for k in range(M):
            mean = dayahead_ele_load[t+k]
            standard_deviation = mean * 0.1
            cur = standard_deviation * np.random.randn() + mean
            dayin_ele_load[k] = cur
        print('dayin_ele_load:',dayin_ele_load)


        #冷负荷
        np.random.seed(0)
        dayin_cold_load = np.zeros(M, dtype=np.int32)
        for k in range(M):
            mean = dayahead_cold_load[t+k]
            standard_deviation = mean * 0.1
            cur = standard_deviation * np.random.randn() + mean
            dayin_cold_load[k] = cur
        print('dayin_cold_load:',dayin_cold_load)


        #构建目标方程
        def cost_min(x):  #x为 n*M列表
        # def demo_func(x):
            global dayin_gasturbine_ele,dayin_heatpump,dayin_elechiller,dayin_absorpchiller,dayin_grid,cost_min
            # cost_min = 0
        # #     dayin_gasturbine_t_i = x[0]
        # #     dayin_absorpchiller_t_i = x[1]
        # #     dayin_elechiller_t_i = x[2]
        # #     dayin_heatpump_t_i = x[3]
        # #     dayin_grid_t_i = x[4]
            dayin_gasturbine_ele = []  #0
            dayin_gasturbine_heat = []
            dayin_gasturbine_heat_in = []
            dayin_absorpchiller = []    #1
            dayin_elechiller = []   #2
            dayin_heatpump = []     #3
            dayin_grid = []     #4
            # dayin_coldtank = []   #watertank 为正 表示释能； 为负 表示蓄能
            for m in range(len(x)):
                if m % n == 0:
                    dayin_gasturbine_ele.append(x[m])    # 发电量
                    # dayin_gasturbine[]
                elif m % n == 1:
                    dayin_absorpchiller.append(x[m])
                elif m % n == 2:
                    dayin_elechiller.append(x[m])
                elif m % n == 3:
                    dayin_heatpump.append(x[m])
                else:
                    dayin_grid.append(x[m])

        # 燃料成本
            for gt_ele in dayin_gasturbine_ele:
                gt_heat = GT.GasTurbine(1500).get_heat_in(gt_ele)[1]    #假设GT额定功率为1500kw
                dayin_gasturbine_heat.append(gt_heat)
                gt_heat_in = GT.GasTurbine(1500).get_heat_in(gt_ele)[0]
                dayin_gasturbine_heat_in.append(gt_heat_in)
            fuel_consume = sum(dayin_gasturbine_heat_in) * eco_pr.delttime_dayin  / eco_pr.heatvalue  # kwh/(Kwh# /m³) = m³
            fuel_cost = fuel_consume * eco_pr.fuel_price #元


        # 运维成本

            maintain_cost = eco_pr.cost_gasturbine * eco_pr.delttime_dayin * sum(dayin_gasturbine_ele) +\
                       eco_pr.cost_absorptionchiller * eco_pr.delttime_dayin * sum(dayin_absorpchiller) +\
                       eco_pr.cost_elechiller * eco_pr.delttime_dayin * sum(dayin_elechiller) +\
                       eco_pr.cost_heatpump * eco_pr.delttime_dayin * sum(dayin_heatpump)
                # cost_gasturbine
                # cost_heatpump
                # cost_chiller
                # cost_coldtank


        # 购电成本
            grid_cost = 0
            for k in range(M):
                grid_cost += eco_pr.get_ele_price(t+k)  * eco_pr.delttime_dayin * dayin_grid[k]  # 改 电价会发生变化

        # 总成本
            total_cost_min = fuel_cost + maintain_cost + grid_cost
            return total_cost_min

        def penalty_cold_balance():
            # 冷负荷平衡
            p_cold_balance = 0
            for k in range(M):
                p_cold_balance += (dayin_absorpchiller[k] + dayin_elechiller[k] + dayin_heatpump[k] -dayin_cold_load[k])**2
            return p_cold_balance

        def penalty_ele_balance():
            # 电负荷平衡
            p_ele_balance = 0
            for k in range(M):
                p_ele_balance += (dayin_gasturbine_ele[k] + dayin_grid[k] - Mag_Elechiller.Magnetic_EleChiller2.get_ele_in(dayin_elechiller[k])
                                   - HeatPump.VariaFrequency_EleChiller2.get_ele_in(dayin_heatpump[k])-dayin_ele_load[k])**2
            return p_ele_balance



        def demo_func(x):
            m = 10000
            return cost_min(x) + m * (penalty_cold_balance() + penalty_ele_balance())


        # # def fluc_min(x,t):
        # #     dayin_gasturbine_t_i = x[0]
        # #     dayin_absorpchiller_t_i = x[1]
        # #     dayin_elechiller_t_i = x[2]
        # #     dayin_heatpump_t_i = x[3]
        # #     dayin_grid_t_i = x[4]
        # #     #dayin_coldtank_i = x[3]   #watertank 为正 表示释能； 为负 表示蓄能
        # #     fluc_squa = (dayahead_gasturbine[t] - dayin_gasturbine_t_i) ** 2 + \
        # #                 (dayahead_absorpchiller[t] - dayin_absorpchiller_t_i) ** 2 + \
        # #                 (dayahead_elechiller[t] - dayin_elechiller_t_i) ** 2 + \
        # #                 (dayahead_heatpump[t] - dayin_heatpump_t_i) ** 2 + \
        # #                 (dayahead_grid[t] - dayin_grid_t_i) ** 2
        # #     return fluc_squa
        #
        # def penalty_demand_balance(x,t,i):
        #     p_cold_balance = abs(x[1]+x[2]+x[3]-dayin_cold_load[4*t+i])
        #     p_ele_balance = abs(x[0]+x[4]-dayin_ele_load[4*t+i])    # 连接设备模型！！！  加电制冷等耗电
        #     return p_cold_balance + p_ele_balance
        #
        # def demo_func(x):
        #     m = 1000
        #     return fluc_min(x,t) + m * penalty_demand_balance(x,t,i)

        # !!!!

        constraint_ueq_list = []

        # absorption chiller 受 gas turbine出力约束
        def constraint_absorption_chiller(x):
            #gasturbine heat out - max absorption chiller cold out
            dayin_gasturbine_ele = []  #0
            dayin_absorpchiller = []     # absorption chiller cold out
            dayin_gasturbine_heat = []
            dayin_absorphiller_cold_out_max = []
            dayin_absorphiller_cold_out_constraint = []
            # dayin_coldtank = []   #watertank 为正 表示释能； 为负 表示蓄能
            for m in range(len(x)):
                if m % n == 0:
                    dayin_gasturbine_ele.append(x[m])    # 发电量
                elif m % n == 1:
                    dayin_absorpchiller.append(x[m])
            for k in range(M):
                gt_heat_out_t = GT.GasTurbine(1500).get_heat_in(dayin_gasturbine_ele[k])[1]    #假设GT额定功率为1500kw
                dayin_gasturbine_heat.append(gt_heat_out_t)
                dayin_absorphiller_cold_out_max_t = Abchiller.AbsorptionChiller1200.get_cold_out(dayin_gasturbine_heat[k])
                dayin_absorphiller_cold_out_max.append(dayin_absorphiller_cold_out_max_t)
                dayin_absorphiller_cold_out_constraint_t = dayin_absorpchiller[k] - dayin_absorphiller_cold_out_max_t
                dayin_absorphiller_cold_out_constraint.append(dayin_absorphiller_cold_out_constraint_t)
                return dayin_absorphiller_cold_out_constraint_t
            # return 0

        constraint_ueq_list.append(lambda x,t=t: constraint_absorption_chiller(x))   #容量上限
        # #  watertank 容量上下限约束
        # constraint_ueq_list.append(lambda x: WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0)-1500)   #容量上限
        # constraint_ueq_list.append(lambda x: -WaterTank.ColdStorage1.get_ColdStorage(x[3],coldstorage_t0))   #容量下限

        # watertank 时间终点容量 = 时间起点容量 约束
        #？？？ 现在是逐时优化，如果要确保等式成立，需要将所有时间设备出力作为一维变量同时优化？

        # gasturbine 爬坡约束
        #def constraint_gasturbine(x):
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

        # best_x_list = []
        # best_y_list = []
        # dayin_ele_load_t = []
        # dayin_cold_load_t = []
        # dayin_ele_supply_t = []
        # dayin_cold_supply_t = []



        # coldstorage_t0 = 0
        # coldstorage_start = coldstorage_t0
        # dayin_gasturbine = np.zeros(24*4,dtype=np.int32)
        # dayin_absorpchiller = np.zeros(24*4,dtype=np.int32)
        # dayin_elechiller = np.zeros(24*4,dtype=np.int32)
        # dayin_heatpump = np.zeros(24*4,dtype=np.int32)
        # dayin_grid = np.zeros(24*4,dtype=np.int32)

        # dayin_watertank = np.zeros(24*4,dtype=np.int32)
        # dayin_watertank_storage = np.zeros(24*4,dtype=np.int32)

        #for t in range(24):
        # t=10
        # for i in range(4):
            #global best_x_list,best_y_list
            # constraint_ueq_list = []
            # # 启停约束
            # # constraint_ueq_list.append(lambda x, t=t: x[0] - 0.1 if dayahead_gasturbine[t] == 0 else x[0] - 1500)
            # # constraint_ueq_list.append(lambda x, t=t: x[1] - 0.1 if dayahead_absorpchiller[t] == 0 else x[1] - 1500)
            # # constraint_ueq_list.append(lambda x, t=t: x[2] - 0.1 if dayahead_elechiller[t] == 0 else x[2] - 800)
            # # constraint_ueq_list.append(lambda x, t=t: x[3] - 0.1 if dayahead_heatpump[t] == 0 else x[3] - 500)
            # # constraint_ueq_list.append(lambda x, t=t: x[4] - 0.1 if dayahead_grid[t] == 0 else x[4] - 1500)
            # # 爬坡约束
            # # 偏差约束
            # for k in range(M):
            # constraint_ueq_list.append(lambda x, t=t: abs(x[0] - dayahead_gasturbine[t]) - 200)
            # constraint_ueq_list.append(lambda x, t=t: abs(x[1] - dayahead_absorpchiller[t]) - 200)
            # constraint_ueq_list.append(lambda x, t=t: abs(x[2] - dayahead_elechiller[t]) - 200)
            # constraint_ueq_list.append(lambda x, t=t: abs(x[3] - dayahead_heatpump[t]) - 200)
            # constraint_ueq_list.append(lambda x, t=t: abs(x[4] - dayahead_grid[t]) - 200)
            #
        pso = PSO(func=demo_func, n_dim=n*M, pop=2000, max_iter=80, lb=[0,0,0,0,0]*M, ub=[1500,1500,800,500,2000]*M,
                      constraint_ueq=tuple(constraint_ueq_list), verbose=1)     #[gasturbine,absorpchiller,elechiller,heatpump,grid]
            # 假设watertank 释能和蓄能的max power 相同；均为[0,1500]
        pso.record_mode = True
        pso.run()
        print(len(pso.gbest_x))
        # 画每代pso的成本迭代曲线


        # 输出结果(结果出力)

        # 各设备逐时运行情况
        dayin_gasturbine_ele_r = []  #0
        dayin_gasturbine_heat_r = []
        # dayin_gasturbine_heat_in_r = []
        dayin_absorpchiller_r = []    #1
        dayin_elechiller_r = []   #2
        dayin_heatpump_r = []     #3
        dayin_grid_r = []     #4
        # dayin_coldtank = []   #watertank 为正 表示释能； 为负 表示蓄能
        for m in range(len(pso.gbest_x)):
            if m % n == 0:
                dayin_gasturbine_ele_r.append(pso.gbest_x[m])    # 发电量
                # dayin_gasturbine[]
            elif m % n == 1:
                dayin_absorpchiller_r.append(pso.gbest_x[m])
            elif m % n == 2:
                dayin_elechiller_r.append(pso.gbest_x[m])
            elif m % n == 3:
                dayin_heatpump_r.append(pso.gbest_x[m])
            else:
                dayin_grid_r.append(pso.gbest_x[m])
        for gt_ele in dayin_gasturbine_ele_r:
            gt_heat_r = GT.GasTurbine(1500).get_heat_in(gt_ele)[1]    #假设GT额定功率为1500kw
            dayin_gasturbine_heat_r.append(gt_heat_r)

        print('dayin_gasturbine_ele:',dayin_gasturbine_ele_r)
        print('dayin_gasturbine_heat:',dayin_gasturbine_heat_r)
        # print(dayin_gasturbine_heat_in)
        print('dayin_absorpchiller:',dayin_absorpchiller_r)    #1
        print('dayin_elechiller:',dayin_elechiller_r)   #2
        print('dayin_heatpump:',dayin_heatpump_r)     #3
        print('dayin_grid:',dayin_grid_r)


        # # 画出设备出力曲线
        # t_list = []
        # for i in range(M):
        #     t_list.append(t+i)
        # plt.step(t_list, dayin_gasturbine_ele_r, color='black')
        # plt.step(t_list, dayin_grid_r, color='yellow')
        # plt.show()

        # # plt.step(t_list, dayin_gasturbine_heat_r, color='red')
        # plt.step(t_list, dayin_absorpchiller_r, color='blue')
        # plt.step(t_list, dayin_elechiller_r, color='green')
        # plt.step(t_list,dayin_heatpump_r, color='purple')
        # plt.show()

        # 日内逐时负荷情况
        print('t:',t,'M:',M,'grid price:',eco_pr.get_ele_price(t))
        print('dayahead_ele_load:',dayahead_ele_load[t:t+M])
        print('dayahead_cold_load:',dayahead_cold_load[t:t+M])

        print('dayin_ele_load:',dayin_ele_load)
        print('dayin_cold_load:',dayin_cold_load)
        #检验设备出力是否满足负荷大小
        dayin_ele_out = []
        dayin_cold_out = []
        for k in range(M):
            dayin_ele_out_t = dayin_gasturbine_ele_r[k] + dayin_grid_r[k] - HeatPump.VariaFrequency_EleChiller2.get_ele_in(dayin_heatpump_r[k])\
                                -Mag_Elechiller.Magnetic_EleChiller2.get_ele_in(dayin_elechiller_r[k])
            dayin_ele_out.append(dayin_ele_out_t)
            # dayin_cold_out_t = dayin_gasturbine_heat_r[k]+dayin_absorpchiller_r[k]+dayin_elechiller_r[k]+dayin_heatpump_r[k]
            dayin_cold_out_t = dayin_absorpchiller_r[k]+dayin_elechiller_r[k]+dayin_heatpump_r[k]
            dayin_cold_out.append(dayin_cold_out_t)
        print('dayin_ele_out:',dayin_ele_out)
        print('dayin_cold_out:',dayin_cold_out)


        # 输出下一时刻的运行成本
        # 燃料成本
        gt_heat_in_r = GT.GasTurbine(1500).get_heat_in(dayin_gasturbine_ele_r[0])[0]
        fuel_consume = gt_heat_in_r * eco_pr.delttime_dayin  / eco_pr.heatvalue  # kwh/(Kwh# /m³) = m³
        fuel_cost = fuel_consume * eco_pr.fuel_price #元

        # 运维成本
        maintain_cost = eco_pr.cost_gasturbine * eco_pr.delttime_dayin * dayin_gasturbine_ele_r[0] +\
                       eco_pr.cost_absorptionchiller * eco_pr.delttime_dayin * dayin_absorpchiller_r[0] +\
                       eco_pr.cost_elechiller * eco_pr.delttime_dayin * dayin_elechiller_r[0] +\
                       eco_pr.cost_heatpump * eco_pr.delttime_dayin * dayin_heatpump_r[0]

        # 购电成本
        grid_cost = eco_pr.get_ele_price(t) * eco_pr.delttime_dayin * dayin_grid_r[0]  # 改 电价会发生变化

        # 总成本
        total_cost_min_m = fuel_cost + maintain_cost + grid_cost
        print('total_cost_min_next_t:',total_cost_min_m)
        print('total_cost:',cost_min(pso.gbest_x))
        total_cost_m_t[t] = total_cost_min_m
        # print(dayin_cold_load)

        # 24h设备出力情况更新
        total_gasturbine_m_t[t] = dayin_gasturbine_ele_r[0]
        total_absorpchiller_m_t[t] = dayin_absorpchiller_r[0]
        total_elechiller_m_t[t] = dayin_elechiller_r[0]
        total_heatpump_m_t[t] = dayin_heatpump_r[0]
        total_grid_m_t[t] = dayin_grid_r[0]



    print('total_cost_m_t',total_cost_m_t)   # t=10时刻，不同M情况下下一时刻的成本
    print('total_gasturbine_m_t',total_gasturbine_m_t)
    print('total_absorpchiller_m_t',total_absorpchiller_m_t)
    print('total_elechiller_m_t',total_elechiller_m_t)
    print('total_heatpump_m_t',total_heatpump_m_t)
    print('total_grid_m_t',total_grid_m_t)


    # # 储存下各时刻设备的出力情况 & watertank_storage
    # dayin_gasturbine[4*t+i] = pso.gbest_x[0]
    # dayin_absorpchiller[4*t+i] = pso.gbest_x[1]
    # dayin_elechiller[4*t+i] = pso.gbest_x[2]
    # dayin_heatpump[4 * t + i] = pso.gbest_x[3]
    # dayin_grid[4 * t + i] = pso.gbest_x[4]
    # # dayin_watertank[4*t+i] = pso.gbest_x[3]   #watertank 为正 表示释能； 为负 表示蓄能
    # # coldstorage_t0 = WaterTank.ColdStorage1.get_ColdStorage(dayin_watertank[4*t+i],coldstorage_t0)
    # # dayin_watertank_storage[4*t+i] = coldstorage_t0

# print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
# print(cost_min(pso.gbest_x), penalty_cold_balance(),penalty_ele_balance())
# dayin_ele_load_t.append(dayin_ele_load[4 * t + i])
# dayin_cold_load_t.append(dayin_cold_load[4 * t + i])
# dayin_ele_supply_t.append(sum(pso.gbest_x[i] for i in (0,4)))
# dayin_cold_supply_t.append(sum(pso.gbest_x[i] for i in (1,2,3)))
# best_x_list.append(list(pso.gbest_x))
# best_y_list.append(list(pso.gbest_y))
#
# print('10h_dayahead_ele_load:',dayahead_ele_load[10])
# print('10h_dayahead_cold_load:',dayahead_cold_load[10])
# print('10h_dayahead_power:',dayahead_gasturbine[10
# ],dayahead_absorpchiller[10],
#       dayahead_elechiller[10],dayahead_heatpump[10],dayahead_grid[10])  #!!
# # print('10h_dayahead_power:',dayahead_gasturbine[10],dayahead_heatpump[10],dayahead_chiller[10],dayahead_watertank[10])  #!!
# print('10h_dayin_power:',best_x_list)
# print('demo_func_min:',best_y_list)
# print('dayin_ele_load_t:',dayin_ele_load_t)
# print('dayin_ele_supply_t:',dayin_ele_supply_t)
# print('dayin_cold_load_t:',dayin_cold_load_t)
# print('dayin_cold_supply_t:',dayin_cold_supply_t)


# print(fluc_min(pso.gbest_x))
# print(penalty_cold_balance(pso.gbest_x))
# print(dayahead_gasturbine[10],dayahead_heatpump[10],dayahead_chiller[10])
# print(dayahead_gasturbine[10]-pso.gbest_x[0],dayahead_heatpump[10]-pso.gbest_x[1],dayahead_chiller[10]-pso.gbest_x[2])
