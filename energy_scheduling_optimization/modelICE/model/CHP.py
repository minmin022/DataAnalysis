from modelICE.model import GasTurbine
from modelICE.model import AbsorptionChiller

gasturbine_heat_out_effi = 0.9


class CHP:
    def __init__(self,CHP_ele_out):
        self.CHP_ele_out = CHP_ele_out
        self.gasturbine_heat_in = GasTurbine.GasTurbine600.get_heat_in(CHP_ele_out)[0]
        self.gasturbine_heat_out = GasTurbine.GasTurbine600.get_heat_in(CHP_ele_out)[1]

    def get_cold_out(self):
        if AbsorptionChiller.AbsorptionChiller2460.get_cold_out(self.gasturbine_heat_out * gasturbine_heat_out_effi)\
            > AbsorptionChiller.nominal_AbsorptionChiller:
            CHP_cold_out = AbsorptionChiller.nominal_AbsorptionChiller
        else:
            CHP_cold_out = AbsorptionChiller.AbsorptionChiller2460.get_cold_out(self.gasturbine_heat_out * gasturbine_heat_out_effi)
        return CHP_cold_out