import os
import tempfile
from subprocess import check_output

from environments import VRPSolver
from instances import VRPSolution, Route, VRPInstance
from utils.vrp_io import write_vrp, read_solution, read_vrp


class LKHSolver(VRPSolver):
    def __init__(self, executable: str):
        super().__init__("lkh")
        self.executable = executable

    def reset(self, instance: VRPInstance):
        self.instance = instance

    def solve(self, instance: VRPInstance, max_steps=None, time_limit=None):
        # assert time_limit is None, "LKH3 does not provide any time limitation parameter"
        self.reset(instance)
        if max_steps is None:
            max_steps = self.instance.n_customers
        with tempfile.TemporaryDirectory() as tempdir:
            problem_filename = os.path.join(tempdir, "problem.vrp")
            output_filename = os.path.join(tempdir, "output.tour")
            param_filename = os.path.join(tempdir, "params.par")

            write_vrp(self.instance, problem_filename)
            params = {"PROBLEM_FILE": problem_filename,
                      "OUTPUT_TOUR_FILE": output_filename,
                      "MAX_TRIALS": max_steps}
            self.write_lkh_par(param_filename, params)
            check_output([self.executable, param_filename])
            tours = read_solution(output_filename, self.instance.n_customers)
            tours = [Route(t, self.instance) for t in tours]
            self.solution = VRPSolution(self.instance, tours)
            return self.solution

    @staticmethod
    def write_lkh_par(filename, parameters):
        default_parameters = {  # Use none to include as flag instead of kv
            "SPECIAL": None,
            "RUNS": 10,
            "TRACE_LEVEL": 1,
            "SEED": 0
        }
        with open(filename, 'w') as f:
            for k, v in {**default_parameters, **parameters}.items():
                if v is None:
                    f.write("{}\n".format(k))
                else:
                    f.write("{} = {}\n".format(k, v))


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    lkh_solver = LKHSolver("../../executables/LKH")
    lkh_solver.solve(inst)
    lkh_solver.render()
