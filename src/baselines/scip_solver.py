import re
from matplotlib import pyplot as plt

from environments import VRPSolver
from instances import VRPInstance, VRPSolution
from instances.vrp_model_scip import VRPModelSCIP
from utils.vrp_io import read_vrp


class SCIPSolver(VRPSolver):
    def __init__(self, lns_only=False):
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
        assignment = {var.name: self.model.getVal(var) for var in self.model.getVars() if 'x' in var.name}
        edges = [tuple([int(n) for n in re.findall(r'\d+', name)])
                 for name, val in assignment.items() if val > 0.99]
        self.solution = VRPSolution.from_edges(instance=self.instance, edges=edges)
        return self.solution


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    scipsolver = SCIPSolver(lns_only=False)
    sol = scipsolver.solve(inst, time_limit=10)
    inst.plot(sol)
    plt.show()
