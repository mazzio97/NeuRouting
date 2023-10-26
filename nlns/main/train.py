import argparse
import torch

from nlns.utils.logging import MultipleLogger, ConsoleLogger, WandBLogger
from nlns.operators.neural import NLNSTrainer
from nlns.operators.destroy import PointDestroy, TourDestroy, RandomDestroy
from nlns.operators.destroy.heatmap import HeatmapDestroy
from nlns.operators.repair import GreedyRepair
from nlns.operators.repair.rl_agent import RLAgentRepair
from nlns.models import VRPActorModel, VRPCriticModel
from nlns.instances import generate_instances


def main(args: argparse.Namespace):
    if args.seed is not None:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger = MultipleLogger(loggers=[ConsoleLogger()])
    if args.wandb_name is not None:
        logger.add(WandBLogger(args.wandb_name))

    destroy_operator_map = {
        'point': lambda: PointDestroy(args.destroy_percentage),
        'tour': lambda: TourDestroy(args.destroy_percentage),
        'random': lambda: RandomDestroy(args.destroy_percentage),
        'heatmap': lambda: HeatmapDestroy.from_checkpoint(
            args.destroy_percentage, 'resgatedgcn100.zip',
            device=device)
    }
    destroy_operator = destroy_operator_map[args.destroy]()
    destroy_operator.set_random_state(args.seed)

    repair_operator_map = {
        # "scip": lambda: SCIPRepair(),
        'greedy': lambda: GreedyRepair(),
        'neural': lambda: RLAgentRepair(
            VRPActorModel(device=device), VRPCriticModel(), device=device,
            logger=logger)
    }
    repair_operator = repair_operator_map[args.repair]()
    repair_operator.set_random_state(args.seed)

    valid_seed = None
    if args.seed is not None:
        valid_seed = args.seed + 1
    dataset = tuple(generate_instances(args.train_samples, args.n_customers,
                                       seed=args.seed))
    validation = tuple(generate_instances(args.val_samples, args.n_customers,
                                          seed=valid_seed))

    trainer = NLNSTrainer(destroy_operator, repair_operator)
    trainer.train(dataset,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  validation=validation,
                  val_interval=args.val_interval,
                  val_batch_size=args.validation_batch_size,
                  log_interval=args.log_interval,
                  logger=logger)

    if trainer.destroy_trainable:
        destroy_operator.save(args.destroy_path, args.epochs - 1, -1)

    if trainer.repair_trainable:
        repair_operator.save(args.repair_path, args.epochs - 1, -1)


def int_or_none(arg):
    try:
        return int(arg)
    except TypeError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Neural VRP')
    parser.add_argument('-d', '--destroy', type=str, required=True,
                        choices=['point', 'tour', 'random', 'neural',
                                 'heatmap'])
    parser.add_argument('--destroy-path', type=str, required=False)
    parser.add_argument('-r', '--repair', type=str, required=True,
                        choices=['greedy', 'scip', 'neural'])
    parser.add_argument('--repair-path', type=str, required=False)
    parser.add_argument('-n', '--n-customers', type=int, required=True)
    parser.add_argument('-p', '--destroy-percentage', type=float,
                        required=True)
    parser.add_argument('-ts', '--train-samples', type=int, default=100000)
    parser.add_argument('-vs', '--val-samples', type=int, default=100)
    parser.add_argument('-bs', '--batch-size', type=int, default=256)
    parser.add_argument('-val-bs', '--validation-batch-size', type=int,
                        default=256)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-vi', '--val-interval', type=int, required=False)
    parser.add_argument('-log', '--log-interval', type=int, required=False)
    parser.add_argument('--wandb-name', type=str, required=False, default=None)
    parser.add_argument('--seed', type=int_or_none, required=False,
                        default=None)

    args = parser.parse_args()

    main(args)
