import os
import re
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt

import nlns
from nlns.instances.vrp_solution import VRPInstance, VRPSolution
from nlns.operators import RepairProcedure

# Optional dependencies come later for practical reasons
nlns.module_found('pyscipopt', __name__)
from pyscipopt import Model                                 # NOQA
from nlns.baselines.scip_solver import VRPModelSCIP         # NOQA


class SCIPRepair(RepairProcedure):

    def __init__(self, time_limit: int = 10):
        self.time_limit = time_limit

    @staticmethod
    def get_solution_edges(model: Model) -> List[tuple]:
        assignment = {var.name: model.getVal(var) for var in model.getVars() if 'x' in var.name}
        return [tuple([int(n) for n in re.findall(r'\d+', name)])
                for name, val in assignment.items() if val > 0.99]

    def __call__(self, partial_solutions: List[VRPSolution]):
        if len(partial_solutions) == 1:
            results = [self._repair(partial_solutions[0])]
        else:
            with Pool(os.cpu_count()) as pool:
                results = pool.map(self._repair, partial_solutions)
                pool.close()
                pool.join()
        for sol, res in zip(partial_solutions, results):
            sol.routes = res.routes

        return partial_solutions

    def _repair(self, partial_solution: VRPSolution):
        model = VRPModelSCIP(partial_solution.instance, lns_only=True)
        model.hideOutput()
        sub_mip = Model(sourceModel=model, origcopy=True)
        varname2var = {v.name: v for v in sub_mip.getVars() if 'x' in v.name}
        for x, y in partial_solution.as_edges():
            sub_mip.fixVar(varname2var[f'x({x}, {y})'], 1)
        sub_mip.setParam("limits/time", self.time_limit)
        sub_mip.optimize()
        new_sol = VRPSolution.from_edges(partial_solution.instance, self.get_solution_edges(sub_mip))
        partial_solution.routes = new_sol.routes
        return new_sol
