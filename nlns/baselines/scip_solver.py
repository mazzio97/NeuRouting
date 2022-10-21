from matplotlib import pyplot as plt

import nlns.generators
from nlns.environments import VRPSolver
from nlns.instances import VRPInstance, VRPSolution
from nlns.instances.vrp_model_scip import VRPModelSCIP


class SCIPSolver(VRPSolver):
    def __init__(self, lns_only=True):
        super().__init__("scip")
        self.model = None
        self.lns_only = lns_only

    def reset(self, instance: VRPInstance):
        self.instance = instance
        self.model = VRPModelSCIP(instance, self.lns_only)

    def solve(self, instance: VRPInstance, time_limit: int, max_steps=None) -> VRPSolution:
        # assert max_steps is None, "SCIP does not provide any max iterations parameter"
        self.reset(instance)
        self.model.setParam("limits/time", time_limit)
        self.model.optimize()
        best_sol = self.model.getBestSol()
        assignment = {var.name: self.model.getSolVal(best_sol, var) for var in self.model.getVars() if 'x' in var.name}
        edges_vars = [name for name, val in assignment.items() if val > 0.99]
        edges = self.model.vars_to_edges(edges_vars)
        self.solution = VRPSolution.from_edges(instance=self.instance, edges=edges)
        return self.solution


if __name__ == "__main__":
    inst = generators.generate_instance(n_customers=50)
    scipsolver = SCIPSolver()
    sol = scipsolver.solve(inst, time_limit=180)
    sol.plot()
    plt.show()
