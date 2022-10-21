import argparse
import sys
import torch
import numpy as np

sys.path.append("src")

from utils.logging import MultipleLogger, TabularConsoleLogger, WandBLogger
from nlns.neural import NeuralProcedurePair
from nlns.destroy import DestroyPointBased, DestroyTourBased, ResGatedGCNDestroy, RandomDestroy
from nlns.repair import SCIPRepair, GreedyRepair
from generators.dataset import NazariDataset

parser = argparse.ArgumentParser(description='Train Neural VRP')
parser.add_argument('-d', '--destroy', type=str, required=True, choices=["point", "tour", "random", "neural"])
parser.add_argument('--destroy_path', type=str, required=False)
parser.add_argument('-r', '--repair', type=str, required=True, choices=["greedy", "scip", "neural"])
parser.add_argument('--repair_path', type=str, required=False)
parser.add_argument('-n', '--n_customers', type=int, required=True)
parser.add_argument('-p', '--destroy_percentage', type=float, required=True)
parser.add_argument('-ts', '--train_samples', type=int, default=100000)
parser.add_argument('-vs', '--val_samples', type=int, default=100)
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-val-bs', '--validation_batch_size', type=int, default=256)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-vi', '--val_interval', type=int, required=False)
parser.add_argument('-log', '--log_interval', type=int, required=False)
parser.add_argument('--seed', type=int, required=False, default=42)

args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert args.destroy == "neural" or args.repair == "neural", "None of the specified operators needs to be trained"
    assert args.destroy != "neural" or args.destroy_path != None, "Define a save path for the destroy operator using --destroy-path"
    assert args.repair != "neural" or args.repair_path != None, "Define a save path for the repair operator using --repair-path"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    destroy_operator_map = {
        "point": lambda: DestroyPointBased(args.destroy_percentage),
        "tour": lambda: DestroyTourBased(args.destroy_percentage),
        "random": lambda: RandomDestroy(args.destroy_percentage),
        "neural": lambda: ResGatedGCNDestroy(args.destroy_percentage, model_config={"device": device})
    }
    destroy_operator = destroy_operator_map[args.destroy]()

    repair_operator_map = {
        "scip": lambda: SCIPRepair(),
        "greedy": lambda: GreedyRepair()
    }
    repair_operator = repair_operator_map[args.repair]()

    dataset = NazariDataset(args.train_samples, args.n_customers)
    # validation dataset is consolidated into a list to always use the same set
    validation = list(NazariDataset(args.val_samples, args.n_customers))

    logger = TabularConsoleLogger(["phase", "epoch", "epoch_step", "target_solution", "incumbent_solution", "time"])

    npp = NeuralProcedurePair(destroy_operator, repair_operator)
    npp.train(dataset, 
              epochs=args.epochs, 
              batch_size=args.batch_size,
              validation=validation,
              val_interval=args.val_interval,
              val_batch_size=args.validation_batch_size,
              log_interval=args.log_interval,
              logger=logger)
    
    if args.destroy == "neural":
        destroy_operator.save(args.destroy_path)
    if args.repair == "neural":
        repair_operator.save(args.repair_path)
