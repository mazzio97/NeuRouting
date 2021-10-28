import re
from pyscipopt import Model, quicksum

from instances import VRPInstance, VRPSolution


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
        u = [self.addVar(vtype="C", name=f"u({i})", lb=1, ub=Q) for i in N]

        # Objective
        self.setObjective(quicksum(x[i, j] * c[i, j] for (i, j) in x), sense='minimize')

        # Constraints
        for i in N:
            self.addCons(quicksum(x[i, j] for j in V if j != i) == 1)

        for j in N:
            self.addCons(quicksum(x[i, j] for i in V if j != i) == 1)

        for (i, j) in x:
            if i != 0 and j != 0:
                self.addCons((u[i - 1] + q[j - 1]) * x[i, j] == u[j - 1] * x[i, j])

        for (i, j) in x:
            if i != 0 and j != 0:
                self.addCons(u[j - 1] >= u[i - 1] + q[j - 1] * x[i, j] - Q * (1 - x[i, j]))

        for i in N:
            self.addCons(u[i - 1] >= q[i - 1])

        self.data = x
        self.varname2var = {v.name: v for v in self.getVars() if 'x' in v.name}

        heuristics = ['alns', 'rins', 'rens', 'dins', 'gins', 'clique', 'lpface', 'crossover', 'mutation',
                      'vbounds', 'trustregion', 'localbranching'] if lns_only else None
        self.select_heuristics(heuristics)

    def select_heuristics(self, heuristics=None):
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

    def set_solution(self, solution: VRPSolution):
        new_sol = self.createPartialSol()
        edges = solution.as_edges()
        for c1, c2 in self.data.keys():
            self.setSolVal(new_sol, self.varname2var[f"x({c1}, {c2})"], 1 if (c1, c2) in edges else 0)
        self.addSol(new_sol)

    @staticmethod
    def vars_to_edges(edges_vars):
        return [tuple([int(n) for n in re.findall(r'\d+', str(edge))]) for edge in edges_vars]
