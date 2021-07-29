from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import Parameters2 as Pr
from modelICE.model import Boiler
from modelICE.model.CHP import CHP

# 日前调度优化 无用 供参考!!!!

class Cost_min:
    def __init__(self, x):
        self.x = x

        CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, \
        Boiler_heat_out, Grid_ele = self.x
        global CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, Boiler_heat_out, Grid_ele
        CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, Boiler_heat_out, Grid_ele = self.x

    def fuel_cost(self):
        global CHP_cold_out, CHP_ele_out, Magnetic_EleChiller_cold_out, Boiler_heat_out, Grid_ele
        fuel_cost = []
        fuel_cost_total = 0
        for t in range(0,24,eco_pr.delttime):
            boiler_fuel = 3600 * Boiler.Boiler1400.get_heat_in(Boiler_heat_out[t]) / eco_pr.heatvalue #m3/h
            gasturbine_fuel = 3600 * CHP(CHP_ele_out[t]).gasturbine_heat_in/ eco_pr.heatvalue #m3/h
            fuel_cost_t = (boiler_fuel + gasturbine_fuel) * eco_pr.fuel_price * eco_pr.delttime   # 时间间隔为1h下的燃气费
            fuel_cost.append(fuel_cost_t)
            fuel_cost_total += fuel_cost_t
            return fuel_cost_total, fuel_cost


    def grid_cost(self):
        grid_cost = []
        grid_cost_total = 0
        for t in range(0,24,eco_pr.delttime):
            grid_cost_t = Grid_ele[t] * eco_pr.get_ele_price(t)
            grid_cost.append(grid_cost_t)
            grid_cost_total += grid_cost_t
        return grid_cost_total, grid_cost


    def maintain_cost(self):
        maintain_cost = []
        maintain_cost_total = 0
        for t in range(0,24,eco_pr.delttime):
            maintain_cost_t = CHP_cold_out[t] * eco_pr.cost_AbsorptionChiller_per_kw  \
                + CHP_ele_out[t] * eco_pr.cost_GasTurbine_per_kw \
                + Magnetic_EleChiller_cold_out[t] * eco_pr.cost_Magnetic_EleChiller_per_kw \
                + Boiler_heat_out[t] * eco_pr.cost_Boiler_per_kw
            maintain_cost.append(maintain_cost_t)
            maintain_cost_total += maintain_cost_t
        return maintain_cost_total, maintain_cost



    def cost_min(self):
        cost_min_total = self.fuel_cost()[0] + self.grid_cost()[0] + self.maintain_cost()[0]
        return cost_min_total














