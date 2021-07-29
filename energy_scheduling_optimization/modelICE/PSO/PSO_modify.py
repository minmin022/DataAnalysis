#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from modelICE.PSO.tools import func_transformer
from modelICE.PSO.base import SkoBase
import matplotlib.pyplot as plt

from modelICE.model import GasTurbine as GT
from modelICE.model import Magnetic_EleChiller as Mag_Elechiller
from modelICE.model import AbsorptionChiller as Abchiller
from modelICE.model import VariaFrequency_EleChiller as HeatPump    #用vf_elechiller 的运行曲线模拟heatpump
from modelICE.model import WaterTank as WaterTank
from modelICE.model import EleStorage as battery
from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import demand
from modelICE.Parameters import economy_parameters as eco_pr
from modelICE.Parameters import demand

dayahead_ele_load = demand.dayahead_ele_load
dayahead_cold_load = demand.dayahead_cold_load

def vari_sepa(x,M):
    dayahead_gasturbine_ele = np.zeros(M, dtype=np.int32)  # 0
    dayahead_absorpchiller = np.zeros(M, dtype=np.int32)  # 1
    dayahead_elechiller = np.zeros(M, dtype=np.int32)  # 2
    dayahead_heatpump = np.zeros(M, dtype=np.int32)  # 3
    dayahead_grid = np.zeros(M, dtype=np.int32)  # 4
    dayahead_watertank = np.zeros(M, dtype=np.int32)  # 5  #watertank 为正 表示释能； 为负 表示蓄能
    dayahead_battery = np.zeros(M, dtype=np.int32)  # 6

    for m in range(len(x)):  #7*24
        dayahead_gasturbine_ele = x[0::7]  # 发电量
        dayahead_absorpchiller  = x[1::7]
        dayahead_elechiller     = x[2::7]
        dayahead_heatpump       = x[3::7]
        dayahead_grid           = x[4::7]
        dayahead_watertank      = x[5::7]
        dayahead_battery        = x[6::7]

    return dayahead_gasturbine_ele,dayahead_absorpchiller,dayahead_elechiller,dayahead_heatpump,\
           dayahead_grid,dayahead_watertank,dayahead_battery

def constraint_absorptionchiller(x,k,M):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x,M)

    dayahead_gasturbine_heat = np.zeros(M, dtype=np.int32)
    dayahead_gasturbine_heat_in = np.zeros(M, dtype=np.int32)
    for u in range(len(dayahead_gasturbine_ele)):
        gt_ele = dayahead_gasturbine_ele[u]
        gt_heat = GT.GasTurbine(1500).get_heat_in(gt_ele)[1]  # 假设GT额定功率为1500kw
        dayahead_gasturbine_heat[u] = gt_heat
        gt_heat_in = GT.GasTurbine(1500).get_heat_in(gt_ele)[0]
        dayahead_gasturbine_heat_in[u] = gt_heat_in
    #制冷能力上限
    constraint_capacity = Abchiller.AbsorptionChiller2460.get_ele_in(dayahead_absorpchiller[k]) - dayahead_gasturbine_heat[k]

    return max(0,constraint_capacity)

#watertank  蓄能容量上下限
def constraint_watertank(x,k,M):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x,M)
    watertank_storage = 0
    for i in range(k):
        #制冷能力上限
        watertank_storage = WaterTank.WaterTank1.get_ColdStorage(dayahead_watertank[i], watertank_storage)

    # return -watertank_storage,watertank_storage-2000
    return max(0,abs(watertank_storage-1000)-1000)
#
#battery 蓄能容量上下限
def constraint_battery(x,k,M):
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x,M)
    battery_storage = 0
    for i in range(k):
        #制冷能力上限
        battery_storage =  battery.EleStorage1.get_EleStorage(dayahead_battery[i], battery_storage)
    return max(0, abs(battery_storage-1000)-1000)

def eqc_ele_balance(x,k,M):
    # 电负荷平衡
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x,M)

    ele_balance = dayahead_ele_load[k] - (dayahead_gasturbine_ele[k] + dayahead_grid[k] + dayahead_battery[k]-
                      Mag_Elechiller.Magnetic_EleChiller2.get_ele_in(dayahead_elechiller[k])
                      - HeatPump.VariaFrequency_EleChiller2.get_ele_in(dayahead_heatpump[k]))
    return ele_balance

def eqc_cold_balance(x,k,M):
    # 冷负荷平衡
    (dayahead_gasturbine_ele, dayahead_absorpchiller, dayahead_elechiller,
     dayahead_heatpump, dayahead_grid, dayahead_watertank, dayahead_battery) = vari_sepa(x,M)
    cold_balance = dayahead_cold_load[k] - (dayahead_absorpchiller[k] + dayahead_elechiller[k] + dayahead_heatpump[k] + dayahead_watertank[k])
    return cold_balance

class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5,
                 constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False, T=None
                 , dim=None):

        n_dim = n_dim or dim  # support the earlier version

        self.M = T
        self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim,dtype=np.int32), \
                           np.array(ub) * np.ones(self.n_dim,dtype=np.int32)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim)) # 在lb~ub范围内生成随机粒子大小
        # LB = [800,304,418,0,50,0,21,761,541,105,0,4,65,46,720,206,322,71,70,100,55,767,167,500,46,102,-37,200]
        # UB = [i + 10 for i in LB]
        # self.X =  np.random.uniform(low=LB, high=UB, size=(self.pop, self.n_dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles  #取最大值 infinity
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated

    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_V(self, iter):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = (0.9-(0.9-0.4)*(self.max_iter-iter)/(self.max_iter))* self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)
        # self.V = 0.9*(self.w*self.max_iter/(self.max_iter+iter) * self.V +
        #          self.cp * r1 * (self.pbest_x - self.X) +
        #          self.cg * r2 * (self.gbest_x - self.X))
        # self.V = self.w * self.V + \
        #          self.cp * r1 * (self.pbest_x - self.X) + \
        #          self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        M = int(len(self.X[1]) / 7)
        # dayahead_gasturbine_ele = x[0::7]  # 发电量
        # dayahead_absorpchiller = x[1::7]
        # dayahead_elechiller = x[2::7]
        # dayahead_heatpump = x[3::7]
        # dayahead_grid = x[4::7]
        # dayahead_watertank = x[5::7]
        # dayahead_battery = x[6::7]
        # watertank_storage = 0
        # watertank = x[5::7]
        # self.X维度是300 * 7 * M   300为行， 7M为列
        p_up = self.ub - self.X
        p_down = self.X - self.lb
        p_up[p_up < 0] = 0  # define the upper margin for increasing power output
        p_down[p_down < 0] = 0  # define the lower margin for decreasing power output
        """"""
        # print(M)

        X = self.X
        for i in range(self.pop):
            for k in range(M):
                # cold balance
                if eqc_cold_balance(self.X[i][:], k, self.M) > 0 and \
                        ((p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) > 0):
                    # print('More!!!!!!')
                    # dayahead_absorpchiller[k]
                    X[i][1 + 7 * k] += p_up[i][1 + 7 * k] / (p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + \
                                                             p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_elechiller[k]
                    X[i][2 + 7 * k] += p_up[i][2 + 7 * k] / (p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + \
                                                             p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_heatpump[k]
                    X[i][3 + 7 * k] += p_up[i][3 + 7 * k] / (p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + \
                                                             p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_watertank[k]
                    X[i][5 + 7 * k] += p_up[i][5 + 7 * k] / (p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + \
                                                             p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)

                elif eqc_cold_balance(self.X[i][:], k, self.M) < 0 and \
                        ((p_up[i][1 + 7 * k] + p_up[i][2 + 7 * k] + p_up[i][3 + 7 * k] + p_up[i][5 + 7 * k]) > 0):
                    # print('Less!!!!!!')
                    X[i][1 + 7 * k] += p_down[i][1 + 7 * k] / (p_down[i][1 + 7 * k] + p_down[i][2 + 7 * k] + \
                                                               p_down[i][3 + 7 * k] + p_down[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_elechiller[k]
                    X[i][2 + 7 * k] += p_down[i][2 + 7 * k] / (p_down[i][1 + 7 * k] + p_down[i][2 + 7 * k] + \
                                                               p_down[i][3 + 7 * k] + p_down[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_heatpump[k]
                    X[i][3 + 7 * k] += p_down[i][3 + 7 * k] / (p_down[i][1 + 7 * k] + p_down[i][2 + 7 * k] + \
                                                               p_down[i][3 + 7 * k] + p_down[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)
                    # dayahead_watertank[k]
                    X[i][5 + 7 * k] += p_down[i][5 + 7 * k] / (p_down[i][1 + 7 * k] + p_down[i][2 + 7 * k] + \
                                                               p_down[i][3 + 7 * k] + p_down[i][5 + 7 * k]) * \
                                       eqc_cold_balance(X[i][:], k, self.M)

                # if constraint_absorptionchiller(X[i][:], k, self.M) + constraint_watertank(X[i][:], k, self.M) + \
                #         constraint_battery(X[i][:], k, self.M) < 50:
                #     self.X[i][:] = X[i][:]
                #
                # X = self.X

                # ele balance
                if eqc_ele_balance(self.X[i][:], k, self.M) > 0 and \
                        ((p_up[i][0 + 7 * k] + p_up[i][4 + 7 * k] + p_up[i][6 + 7 * k]) > 0):
                    # dayahead_gasturbine_ele[k]
                    X[i][0 + 7 * k] += p_up[i][0 + 7 * k] / (
                                p_up[i][0 + 7 * k] + p_up[i][4 + 7 * k] + p_up[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)
                    # dayahead_grid[k]
                    X[i][4 + 7 * k] += p_up[i][4 + 7 * k] / (
                                p_up[i][0 + 7 * k] + p_up[i][4 + 7 * k] + p_up[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)
                    # dayahead_battery[k]
                    X[i][6 + 7 * k] += p_up[i][6 + 7 * k] / (
                                p_up[i][0 + 7 * k] + p_up[i][4 + 7 * k] + p_up[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)
                elif eqc_ele_balance(self.X[i][:], k, self.M) < 0 and \
                        ((p_down[i][0 + 7 * k] + p_down[i][4 + 7 * k] + p_down[i][6 + 7 * k]) > 0):
                    X[i][0 + 7 * k] += p_down[i][0 + 7 * k] / (
                                p_down[i][0 + 7 * k] + p_down[i][4 + 7 * k] + p_down[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)
                    # dayahead_elechiller[k]
                    X[i][4 + 7 * k] += p_down[i][4 + 7 * k] / (
                                p_down[i][0 + 7 * k] + p_down[i][4 + 7 * k] + p_down[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)
                    # dayahead_heatpump[k]
                    X[i][6 + 7 * k] += p_down[i][6 + 7 * k] / (
                                p_down[i][0 + 7 * k] + p_down[i][4 + 7 * k] + p_down[i][6 + 7 * k]) * \
                                       eqc_ele_balance(X[i][:], k, self.M)

                # if constraint_absorptionchiller(X[i][:], k, self.M) + constraint_watertank(X[i][:], k, self.M) + \
                #         constraint_battery(X[i][:], k, self.M) < 50:
                #     self.X[i][:] = X[i][:]

                self.X = X

        self.X = np.clip(self.X, self.lb, self.ub)  # 超出了上下界限 直接砍掉 取上下界

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y  #need——update=true 就开始更新
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)  #如果算出来的结果更小，但是违反了约束，还是不会更新pbest

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=10, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        # y_iter = {}
        for iter_num in range(self.max_iter):
            self.update_V(iter_num)
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)  #local max - local min 不怎么变化
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
            if iter_num == self.max_iter - 1:
                print('gbest_y_hist:',self.gbest_y_hist)
                fig1,ax1 = plt.subplots()
                plt.plot([i for i in range(self.max_iter)],self.gbest_y_hist)
                ax1.set_yscale('log')
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run
