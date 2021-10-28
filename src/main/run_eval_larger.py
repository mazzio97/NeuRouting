import argparse
import sys
from typing import List
from pathlib import Path
import torch
from matplotlib import pyplot as plt

from nlns.builder import nlns_builder
from utils.vrp_io import read_vrp

sys.path.append("src")

from baselines import SCIPSolver, ORToolsSolver, LKHSolver
from environments import VRPSolver
from main.evaluator import Evaluator

parser = argparse.ArgumentParser(description='Evaluate Neural VRP')
parser.add_argument('-c', '--n_customers', type=int, required=True)
parser.add_argument('-dm', '--destroy_methods', nargs='+')
parser.add_argument('-dp', '--destroy_percentage', nargs='+')
parser.add_argument('-rm', '--repair_methods', nargs='+')
parser.add_argument('-t', '--time_limit', type=int, default=60)
parser.add_argument('-max', '--max_steps', type=int, required=False)
parser.add_argument('-ns', '--neighborhood_size', type=int, default=256)
parser.add_argument('-r', '--runs', type=int, default=1)
parser.add_argument('-a', '--adaptive', action='store_true', default=False)
parser.add_argument('-b', '--baselines', nargs='+', required=False)
args = parser.parse_args()

if __name__ == "__main__":
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for evaluation.")

    ckpt_path = f"./pretrained/"

    files = [inst for inst in Path('./res/instances/').rglob("*.vrp")]
    eval_instances = [read_vrp(file, grid_dim=1000) for file in files]

    baselines = [SCIPSolver(), ORToolsSolver(), LKHSolver("./executables/LKH")]
    if args.baselines is not None:
        solvers: List[VRPSolver] = [solver for name in args.baselines for solver in baselines if name == solver.name]
    else:
        solvers = []

    if args.destroy_percentage is not None:
        destroy_percentage = [float(percentage) for percentage in args.destroy_percentage]

        if args.adaptive:
            name = 'destroy_' + '+'.join(args.destroy_methods) + "_repair_" + '+'.join(args.repair_methods)
            destroy_names = {method: destroy_percentage for method in args.destroy_methods}
            adaptive_lns_env = nlns_builder(destroy_names=destroy_names,
                                            repair_names=args.repair_methods,
                                            neighborhood_size=args.neighborhood_size,
                                            name=name,
                                            device=device,
                                            n=args.n_customers,
                                            ckpt_path=ckpt_path)
            solvers.append(adaptive_lns_env)
        else:
            for destroy in args.destroy_methods:
                for repair in args.repair_methods:
                    name = 'destroy_' + destroy + "_repair_" + repair + '_p' + '_'.join(args.destroy_percentage)
                    lns_env = nlns_builder(destroy_names={destroy: destroy_percentage},
                                           repair_names=repair,
                                           neighborhood_size=args.neighborhood_size,
                                           name=name,
                                           device=device,
                                           n=args.n_customers,
                                           ckpt_path=ckpt_path)
                    solvers.append(lns_env)

    evaluator = Evaluator(solvers)
    stats = evaluator.compare([eval_instances[0]], n_runs=args.runs,
                              max_steps=args.max_steps, time_limit=args.time_limit, inst_names=[f.name for f in files])
    for solver, df in stats.to_dataframe(inst_names=[f.name for f in files]).items():
        df.to_csv(f"./res/stats/larger/{solver.name}_larger.csv")
