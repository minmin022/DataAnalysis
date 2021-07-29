from modelICE.system_func import cost_min2
from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import Parameters2 as Pr, demand
from modelICE.model import Magnetic_EleChiller as M_EC
from modelICE.model.CHP import CHP


# 日前调度优化 无用 供参考!!!!

class Constraint(cost_min2.Cost_min):
    def __init__(self, x):
        super().__init__(x)
        global CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, Boiler_heat_out, Grid_ele
        CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, Boiler_heat_out, Grid_ele = self.x


# 冷负荷平衡
    def constraint_cooling_demand(self):
        cooling_demand_balance = []
        for t in range(0, 24, eco_pr.delttime):
            cooling_demand_balance_t = CHP_cold_out[t] + Magnetic_EleChiller_cold_out[t] - demand.Q_cooling[t]
        # HeatPump_heat_cold, WaterTank_heat_cold!!!!
            cooling_demand_balance.append(cooling_demand_balance_t)
        return [i-1 for i in cooling_demand_balance], [-i for i in cooling_demand_balance]

# 热负荷平衡
    def constraint_heating_demand(self):
        heating_demand_balance = []
        for t in range(0, 24, eco_pr.delttime):
            heating_demand_balance_t = Boiler_heat_out[t] - demand.Q_heating[t]
        # HeatPump_heat_cold, WaterTank_heat_cold!!!!
            heating_demand_balance.append(heating_demand_balance_t)
        return [i-1 for i in heating_demand_balance], [-i for i in heating_demand_balance]

# 电负荷平衡
    def constraint_ele_demand(self):
        ele_demand_balance = []
        for t in range(0, 24, eco_pr.delttime):
            ele_demand_balance_t = CHP_ele_out[t] + Grid_ele[t] \
                                   - M_EC.Magnetic_EleChiller1.get_ele_in(Magnetic_EleChiller_cold_out[t]) \
                                   - demand.Q_ele[t]
            #还要减去热泵耗电能！！！
            ele_demand_balance.append(ele_demand_balance_t)
        return [i - 1 for i in ele_demand_balance], [-i for i in ele_demand_balance]

# CHP(GasTurbine & AbsorptionChiller)
    def constraint_CHP(self):
        cons_gt_1, cons_gt_2, cons_gt_3 = [], [], []
        for t in range(0, 24, eco_pr.delttime):
            cons_gt_1.append(CHP_ele_out[t] - Pr.P_gasturbine_max)
            cons_gt_2.append(-(CHP_ele_out[t] -Pr.P_gasturbine_min))
            if CHP_ele_out[t] > CHP_ele_out[t-1]:
                cons_gt_3.append(CHP_ele_out[t] - CHP_ele_out[t-1] - Pr.P_gasturbine_up)
            else:
                cons_gt_3.append(CHP_ele_out[t-1] - CHP_ele_out[t] - Pr.P_gasturbine_down)

            CHP_cold_out[t] = CHP(CHP_ele_out[t]).get_cold_out()  # 赋值
        return cons_gt_1, cons_gt_2, cons_gt_3


# Boiler
    def constraint_boiler(self):
        cons_b_1, cons_b_2 = [], []
        for t in range(0, 24, eco_pr.delttime):
            cons_b_1.append(Boiler_heat_out[t] - Pr.P_boiler_max)
            cons_b_2.append(-(Boiler_heat_out[t] -Pr.P_boiler_min))
        return cons_b_1, cons_b_2
      


# EleStorage
#     def constraint_elestorage(self):
#         cons_es_chr_1, cons_es_chr_2 = [], []
#         cons_es_dis_3, cons_es_dis_4 = [], []
#         elestorage = [0]
#         for t in range(0, 24, eco_pr.delttime):
#             if EleStorage_ele[t] >= 0:
#                 u_chr_dis = 1
#                 cons_es_chr_1.append(u_chr_dis * EleStorage_ele[t] - Pr.P_elestorage_chr_max)
#                 cons_es_chr_2.append(-(u_chr_dis * EleStorage_ele[t] - Pr.P_elestorage_chr_min))
#             else: #elif EleStorage_ele[t] < 0:
#                 u_chr_dis = -1
#                 cons_es_dis_3.append(u_chr_dis * EleStorage_ele[t] - Pr.P_elestorage_dis_max)
#                 cons_es_dis_4.append(-(u_chr_dis * EleStorage_ele[t] - Pr.P_elestorage_dis_min))
#             if t > 0:
#                 elestorage.append(EleStorage.EleStorage1.get_EleStorage(elestorage[t-1], EleStorage_ele[t]))
#             elestorage[t]
#
#         return 0

# WaterTank

# HeatPump

# Magnetic_EleChiller
    def constraint_magetic_chiller(self):
        cons_mec_1, cons_mec_2 = [], []
        for t in range(0, 24, eco_pr.delttime):
            cons_mec_1.append(Magnetic_EleChiller_cold_out[t] - Pr.P_Magnetic_EleChiller_max)
            cons_mec_2.append(-(Magnetic_EleChiller_cold_out[t] - Pr.P_Magnetic_EleChiller_min))
        return cons_mec_1, cons_mec_2


# VariaFrequency_EleChiller
#     def constraint_variafrequency_elechiller(self):
#         cons_vfec_1, cons_vfec_2 = [], []
#         for t in range(0, 24, eco_pr.delttime):
#             cons_vfec_1.append(VariaFrequency_EleChiller_cold_out[t] - Pr.P_VariaFrequency_EleChiller_max)
#             cons_vfec_2.append(-(VariaFrequency_EleChiller_cold_out[t] - Pr.P_VariaFrequency_EleChiller_min))
#         return cons_vfec_1, cons_vfec_2