import os
import re
from multiprocessing import Pool
from typing import List

import nlns
from nlns.instances.vrp_solution import VRPSolution
from nlns.operators import LNSOperator

# Optional dependencies come later for practical reasons
nlns.module_found('pyscipopt', __name__)
from pyscipopt import Model                                 # NOQA
from nlns.baselines.scip_solver import VRPModelSCIP         # NOQA


class SCIPRepair(LNSOperator):
    """Repair solutions as a MI problem, using the SCIP opt suite.

    The operator requires ``pyscipopt`` to be imported and used.
    Setup can be tricky, please refer to `the installation
    guide <https://github.com/scipopt/PySCIPOpt#installation>`_.

    Execution happens in parallel, trying to use all cores.

    Args:
        time_limit: the time limit imposed to the MIP solver for each
            instance. See :meth:`__call__` for more info.
    """

    def __init__(self, time_limit: int = 10):
        self.time_limit = time_limit

    @staticmethod
    def get_solution_edges(model: Model) -> List[tuple]:
        """Given a solved model, get an edge representation."""
        assignment = {var.name: model.getVal(var)
                      for var in model.getVars() if 'x' in var.name}
        return [tuple([int(n) for n in re.findall(r'\d+', name)])
                for name, val in assignment.items() if val > 0.99]

    def __call__(self, partial_solutions: List[VRPSolution]
                 ) -> List[VRPSolution]:
        """Run solver on the given solutions.

        Execution time is roughly constrained
        ``time_limit * len(partial_solutions)``.

        Args:
            partial_solutions: The destroyed solutions to be repaired.

        Returns:
            The completely repaired solutions. If the time limit and the
                complexity of the initial solutions permit it, they will
                be the optimal repaired solutions.
        """
        if len(partial_solutions) == 1:
            results = [self.apply_single(partial_solutions[0])]
        else:
            with Pool(os.cpu_count()) as pool:
                results = pool.map(self.apply_single, partial_solutions)

        for sol, res in zip(partial_solutions, results):
            sol.routes = res.routes

        return partial_solutions

    def apply_single(self, partial_solution: VRPSolution) -> VRPSolution:
        """Run solver on one solution.

        Args:
            partial_solution: The destroyed solution to be repaired.

        Returns:
            A completely repaired solution. If the time limit and the
                complexity of the initial solution permit it, it will be
                the optimal repaired solution.
        """
        model = VRPModelSCIP(partial_solution.instance, lns_only=True)
        model.hideOutput()
        sub_mip = Model(sourceModel=model, origcopy=True)
        varname2var = {v.name: v for v in sub_mip.getVars() if 'x' in v.name}
        for x, y in partial_solution.as_edges():
            sub_mip.fixVar(varname2var[f'x({x}, {y})'], 1)
        sub_mip.setParam('limits/time', self.time_limit)
        sub_mip.setParam('randomization/lpseed',
                         self.rng.randint(1, 2147483647))
        sub_mip.optimize()
        new_sol = VRPSolution.from_edges(partial_solution.instance,
                                         self.get_solution_edges(sub_mip))
        partial_solution.routes = new_sol.routes
        return new_sol
