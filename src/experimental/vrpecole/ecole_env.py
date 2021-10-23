import time
from abc import abstractmethod
from typing import Optional

import ecole

from environments import VRPEnvironment
from instances import VRPInstance, VRPSolution, VRPModelSCIP


class VRPInfo:
    def before_reset(self, model: ecole.scip.Model) -> None:
        pass

    def extract(self, model: ecole.scip.Model, done: bool):
        scip_model = model.as_pyscipopt()
        edges_vars = [var for var in scip_model.getVars() if scip_model.getVal(var) > 0.99 and 'x' in str(var)]
        return VRPModelSCIP.vars_to_edges(edges_vars)


class EcoleEnvironment(VRPEnvironment):
    def __init__(self, base_env, name="Ecole"):
        super(EcoleEnvironment, self).__init__(name)
        self.scip_model = None
        self.env = base_env

        self.obs = None
        self.action_set = None
        self.reward = None

    def reset(self, instance: VRPInstance, initial: Optional[VRPSolution] = None):
        super(EcoleEnvironment, self).reset(instance)
        self.scip_model = VRPModelSCIP(instance, lns_only=True)
        if initial is not None:
            self.scip_model.set_solution(initial)
        return self.env.reset(ecole.scip.Model.from_pyscipopt(self.scip_model))

    @abstractmethod
    def step(self):
        pass

    def solve(self, instance: VRPInstance, initial: VRPSolution = None, time_limit=None, max_steps=None) -> VRPSolution:
        self.obs, self.action_set, self.reward, done, edges = self.reset(instance, initial)
        self.solution = VRPSolution.from_edges(self.instance, edges)

        self.max_steps = max_steps if max_steps is not None else self.max_steps
        self.time_limit = time_limit if time_limit is not None else self.time_limit
        start_time = time.time()
        while not done and self.n_steps < self.max_steps and time.time() - start_time < self.time_limit:
            self.obs, self.action_set, self.reward, done, edges = self.step()
            self.solution = VRPSolution.from_edges(self.instance, edges)
            self.n_steps += 1
        return self.solution
