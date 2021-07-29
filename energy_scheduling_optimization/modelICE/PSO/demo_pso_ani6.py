import numpy as np
from modelICE.PSO.PSO import PSO

time = 24
def cost(x):
    maintain_cost = [(2*x[t] + x[t+time]) for t in range(0,time,1)]
    fuel_cost = [(3*x[t] + x[t+time]) for t in range(0,time,1)]
    maintain_cost_total = sum(maintain_cost)
    fuel_cost_total = sum(fuel_cost)
    return maintain_cost_total + fuel_cost_total

def penalty_heat_balance(x):
    p_heat_balance = 0
    for t in range(0,time,1):
        p_heat_balance += abs(x[t]+x[t+time]-Q_heating[t])
    return p_heat_balance

def demo_func(x):
    m = 500
    return cost(x) + m * penalty_heat_balance(x)

Q_heating = np.zeros(time, dtype=np.int32)
for t in range(0, time, 1):
    if t in range(9, 21):
        Q_heating[t] = 1000  # kw
    else:
        Q_heating[t] = 800


max_iter = 1000
lb_range = Q_heating
ub_range = Q_heating + 500

lb = np.zeros(2*time,dtype=np.int32)
ub = np.zeros(2*time, dtype=np.int32)
for t in range(0,time,1):
    lb[t] = 0
    ub[t] = 1500
    lb[t+time] = 0
    ub[t+time] = 800

pso = PSO(func=demo_func, n_dim=48, pop=40, max_iter=max_iter, lb = list(lb),ub = list(ub),verbose = 1)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x.reshape(2,24), 'best_y is', pso.gbest_y)

for t in range(0,time,1):
    print(pso.gbest_x[t]+pso.gbest_x[t+time])


