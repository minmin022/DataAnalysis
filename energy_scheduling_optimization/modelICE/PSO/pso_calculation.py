from modelICE.PSO.PSO import PSO
from modelICE.heating_example.cost_min_heat import Cost_min
from modelICE.heating_example.constraint_heat import Constraint

def func(x):
    return Cost_min(x).cost_min()


constraint_ueq = (
    Constraint.constraint_heating_demand,
    Constraint.constraint_boiler
)

max_iter = 1000
pso = PSO(func=func, n_dim=1, pop=40, max_iter=max_iter, constraint_ueq=constraint_ueq)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
print(pso.gbest_x)