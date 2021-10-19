import re
from builtins import staticmethod

import pyscipopt
from matplotlib import pyplot as plt
from pyscipopt import Model, quicksum

from instances import VRPInstance, VRPSolution
from nlns.initial import nearest_neighbor_solution
from utils.vrp_io import read_vrp


class VRPModelSCIP(Model):
    def __init__(self, instance: VRPInstance, lns_only=False, *args, **kwargs):
        super().__init__('CVRP', *args, **kwargs)
        N = list(range(1, instance.n_customers + 1))
        V = [0] + N
        c = instance.distance_matrix
        Q = instance.capacity
        q = instance.demands

        # Variables
        x = {}
        for i in V:
            for j in V:
                if i != j:
                    x[i, j] = self.addVar(vtype="B", name=f"x({i}, {j})")
        u = [self.addVar(vtype="I", name=f"u({i})") for i in N]

        # Objective
        self.setObjective(quicksum(x[i, j] * c[i, j] for (i, j) in x), sense='minimize')

        # Constraints
        for i in N:
            self.addCons(quicksum(x[i, j] for j in V if j != i) == 1)

        for j in N:
            self.addCons(quicksum(x[i, j] for i in V if j != i) == 1)

        for h in N:
            self.addCons(quicksum(x[i, h] for i in V if i != h) ==
                         quicksum(x[h, i] for i in V if i != h))

        for (i, j) in x:
            if i != 0 and j != 0:
                self.addCons((u[i - 1] + q[j - 1]) * x[i, j] == u[j - 1] * x[i, j])

        for (i, j) in x:
            if i != 0 and j != 0:
                self.addCons(u[j - 1] >= u[i - 1] + q[j - 1] * x[i, j] - Q * (1 - x[i, j]))

        for i in N:
            self.addCons(u[i - 1] >= q[i - 1])

        for i in N:
            self.addCons(u[i - 1] <= Q)

        self.data = x
        self.varname2var = {v.name: v for v in self.getVars() if 'x' in v.name}

        heuristics = ['alns', 'rins', 'rens', 'dins', 'gins', 'clique', 'lpface', 'crossover', 'mutation',
                      'vbounds', 'trustregion', 'localbranching'] if lns_only else None
        self.select_heuristics(heuristics)

    def select_heuristics(self, heuristics=None, seed=42):
        seed = seed % 2147483648  # SCIP seed range

        # set up randomization
        self.setBoolParam('randomization/permutevars', True)
        self.setIntParam('randomization/permutationseed', seed)
        self.setIntParam('randomization/randomseedshift', seed)

        # presolving
        # self.setIntParam('presolving/maxrestarts', 0)
        # self.setIntParam('presolving/maxrounds', 0)

        # disable separating (cuts)
        self.setIntParam('separating/maxroundsroot', 0)
        self.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)

        self.setBoolParam('conflict/enable', False)

        # disable pscost for branching.
        self.setParam('branching/pscost/priority', 1e8)

        self.hideOutput()

        if heuristics is None:
            return

        frequency = {}

        # disable all the heuristics
        for k, v in self.getParams().items():
            if k.startswith('heuristics/') and k.endswith('/freq'):
                self.setParam(k, -1)
                frequency[k.split(sep='/')[1]] = v

        # re-enable only the desired ones
        for h in heuristics:
            self.setParam('heuristics/' + h + '/freq', frequency[h] if frequency[h] > 0 else 1)

    def setSolution(self, solution: VRPSolution):
        new_sol = self.createPartialSol()
        edges = solution.as_edges()
        for c1, c2 in self.data.keys():
            self.setSolVal(new_sol, self.varname2var[f"x({c1}, {c2})"], 1 if (c1, c2) in edges else 0)
        self.addSol(new_sol)

    @staticmethod
    def vars_to_edges(edges_vars):
        return [tuple([int(n) for n in re.findall(r'\d+', str(edge))]) for edge in edges_vars]


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    sol = nearest_neighbor_solution(inst)
    inst.plot(sol)
    plt.show()
    scip_model = VRPModelSCIP(inst)
    # scip_model.setSolution(sol)
    scip_sol = scip_model.getBestSol()
    scip_model.setParam("limits/time", 20)
    scip_model.optimize()
    edges_vars = [var for var in scip_model.getVars() if scip_model.getVal(var) > 0.99 and 'x' in str(var)]
    edges = VRPModelSCIP.vars_to_edges(edges_vars)
    opt_sol = VRPSolution.from_edges(inst, edges)
    inst.plot(opt_sol)
    plt.show()
