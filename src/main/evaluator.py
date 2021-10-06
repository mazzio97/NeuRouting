import time
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from environments import VRPSolver, VRPEnvironment
from instances import VRPInstance


class Stats(Dict[VRPSolver, Dict[VRPInstance, dict]]):
    def __init__(self, solvers: List[VRPSolver]):
        super().__init__([(solver, {}) for solver in solvers])

    def mean_cost(self) -> Dict[str, float]:
        return {solver.name: np.mean([info["cost"] for info in self[solver].values()]) for solver in self.keys()}

    def to_dataframe(self) -> Dict[VRPSolver, pd.DataFrame]:
        dfs = {}
        for solver in self.keys():
            solver_stats = {}
            for i, info in enumerate(self[solver].values()):
                solver_stats[f"instance_{i}"] = info
            dfs[solver] = pd.DataFrame.from_dict(solver_stats, orient='index')
        return dfs


class Evaluator:
    def __init__(self, solvers: List[VRPSolver], render=False):
        self.solvers = solvers
        self.render = render

    def compare(self, instances: List[VRPInstance], max_steps=None, time_limit=None) -> Stats:
        stats = Stats(self.solvers)
        for i, inst in enumerate(instances):
            for j, solver in enumerate(self.solvers):
                stats[solver][inst] = {}
                inst_stats = stats[solver][inst]
                start_time = time.time()
                sol = solver.solve(inst, max_steps=max_steps, time_limit=time_limit)
                solve_time = time.time() - start_time
                sol.verify()
                # inst_stats["solution"] = sol
                inst_stats["n_vehicles"] = len(sol.routes)
                inst_stats["cost"] = sol.cost()
                inst_stats["time"] = solve_time
                if isinstance(solver, VRPEnvironment):
                    inst_stats["steps"] = solver.n_steps
                    inst_stats["improvements"] = solver.improvements
                    inst_stats["gap"] = solver.gap
                print(f"Instance {i} solved by {solver.name} with cost {inst_stats['cost']}")
        return stats
