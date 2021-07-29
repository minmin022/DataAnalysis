from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import Parameters2 as Pr
from modelICE.model import Boiler
from modelICE.model.CHP import CHP
# 日前调度优化 无用 供参考
class Cost_min:
    def __init__(self, x):
        self.x = x
        global Boiler_heat_out
        Boiler_heat_out = self.x


    def fuel_cost(self):
        global Boiler_heat_out
        fuel_cost = []
        fuel_cost_total = 0
        for t in range(0,24,eco_pr.delttime):
            boiler_fuel = 3600 * Boiler.Boiler1400.get_heat_in(Boiler_heat_out[t]) / eco_pr.heatvalue #m3/h
            fuel_cost_t = (boiler_fuel) * eco_pr.fuel_price * eco_pr.delttime   # 时间间隔为1h下的燃气费
            fuel_cost.append(fuel_cost_t)
            fuel_cost_total += fuel_cost_t
            return fuel_cost_total, fuel_cost


    def maintain_cost(self):
        maintain_cost = []
        maintain_cost_total = 0
        for t in range(0,24,eco_pr.delttime):
            maintain_cost_t = Boiler_heat_out[t] * eco_pr.cost_Boiler_per_kw
            maintain_cost.append(maintain_cost_t)
            maintain_cost_total += maintain_cost_t
        return maintain_cost_total, maintain_cost

    def cost_min(self):
        cost_min_total = self.fuel_cost()[0]  + self.maintain_cost()[0]
        return cost_min_total














