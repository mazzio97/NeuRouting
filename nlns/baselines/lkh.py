import os
import sys
import tempfile
from subprocess import check_output
from time import time

import matplotlib.pyplot as plt

from nlns.instances import VRPInstance, generate_instance
from nlns.io import write_vrp, read_solution


class LKHSolver:
    """Setup LKH3 to solve the CVRP problem.

    A binary of LKH must be installed and discoverable by the system
    (e.g. in your PATH). Otherwise, the path to the executable can be
    manually specified through the ``executable`` attribute.
    Source code and instructions on how to compile can be found
    `here <http://webhotel4.ruc.dk/~keld/research/LKH-3/>`_.

    Args:
        executable: The name or path to the LKH executable. By default,
            a binary named ``LKH`` is expected to be resolved by the
            system PATH.
        seed: Seed for reproducibility. Defaults to 0 (random results).
    """

    def __init__(self, executable: str = 'LKH', seed: int = 0):
        # super().__init__("lkh")
        self.executable = executable
        self.seed = seed

    def solve(self, instance: VRPInstance, runs=10, max_trials=None,
              time_limit=sys.float_info.max):
        """Apply the solver to an instance."""
        if max_trials is None:
            max_trials = instance.n_customers

        with tempfile.TemporaryDirectory() as tempdir:
            problem_filename = os.path.join(tempdir, 'problem.vrp')
            output_filename = os.path.join(tempdir, 'output.tour')
            param_filename = os.path.join(tempdir, 'params.par')

            write_vrp(instance, problem_filename)
            params = {'PROBLEM_FILE': problem_filename,
                      'OUTPUT_TOUR_FILE': output_filename,
                      'RUNS': runs,
                      'SEED': self.seed,
                      'TIME_LIMIT': time_limit,
                      'MAX_TRIALS': max_trials}
            self.write_lkh_par(param_filename, params)
            start_t = time()
            check_output([self.executable, param_filename])
            end_t = time()

            solution = read_solution(instance, output_filename)
            solution.time_taken = end_t - start_t

        return solution

    @staticmethod
    def write_lkh_par(filename, input_parameters):
        parameters = {  # Use none to include as flag instead of kv
            'SPECIAL': None,
            'RUNS': 10,
            'TRACE_LEVEL': 1,
            'SEED': 0
        }
        parameters.update(input_parameters)
        with open(filename, 'w') as f:
            for k, v in parameters.items():
                if v is None:
                    f.write(f'{k}\n')
                else:
                    f.write(f'{k} = {v}\n')


if __name__ == '__main__':
    # Demo the solver
    instance = generate_instance(50, seed=1)
    solver = LKHSolver(seed=1)
    sol = solver.solve(instance)
    sol.plot()
    plt.show()
