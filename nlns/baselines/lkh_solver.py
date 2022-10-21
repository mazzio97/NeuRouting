import os
import shutil
import tempfile
from subprocess import check_output

from nlns.environments import VRPSolver
from nlns.instances import VRPSolution, Route, VRPInstance
from nlns.utils.vrp_io import write_vrp, read_solution, read_vrp


class LKHSolver(VRPSolver):
    def __init__(self, executable: str):
        super().__init__("lkh")
        self.executable = executable

    def reset(self, instance: VRPInstance):
        pass

    def initial_solution(self, instance: VRPInstance):
        return self.solve(instance, max_steps=1)

    def solve(self, instance: VRPInstance, max_steps=None, runs=10, time_limit=None):
        # assert time_limit is None, "LKH3 does not provide any time limitation parameter"
        if max_steps is None:
            max_steps = instance.n_customers
        with tempfile.TemporaryDirectory() as tempdir:
            problem_filename = os.path.join(tempdir, "problem.vrp")
            output_filename = os.path.join(tempdir, "output.tour")
            param_filename = os.path.join(tempdir, "params.par")

            write_vrp(instance, problem_filename)
            params = {"PROBLEM_FILE": problem_filename,
                      "OUTPUT_TOUR_FILE": output_filename,
                      "MAX_TRIALS": max_steps}
                      #"RUNS": runs}
            self.write_lkh_par(param_filename, params)
            check_output([self.executable, param_filename])
            tours = read_solution(output_filename, instance.n_customers)
            tours = [Route(t, instance) for t in tours]
            solution = VRPSolution(instance, tours)
        return solution

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
