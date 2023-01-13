import os
import time
from abc import abstractmethod, abstractclassmethod
from math import ceil
from typing import Union, Optional, List, Callable, Tuple, Dict
from more_itertools import chunked
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

from tqdm.auto import tqdm
from nlns.instances import VRPInstance, VRPSolution, VRPNeuralSolution
from nlns.operators import LNSProcedure, RepairProcedure, DestroyProcedure, LNSOperator
from nlns.operators.initial import nearest_neighbor_solution
from nlns.utils.logging import Logger, EmptyLogger


class NeuralProcedure(LNSProcedure):
    def __init__(self, model: nn.Module, device: str = "cpu", logger: Optional[Logger] = None):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.val_env = None
        self._val_phase = False


class NeuralProcedurePair:
    """
    A neural procedure pair is used to train one (or two) neural procedure(s).
    This is done by providing a repair and a destroy procedure.
    Either one or the other procedures needs to be neural based.
    """
    def __init__(self,
                 destroy_procedure: DestroyProcedure,
                 repair_procedure: RepairProcedure):
        assert isinstance(destroy_procedure, NeuralProcedure) or isinstance(repair_procedure, NeuralProcedure), \
            f"Either the destroy procedure or the repair procedure must to be neural based."
        self.destroy_procedure = destroy_procedure
        self.destroy_is_neural = isinstance(self.destroy_procedure, NeuralProcedure)
        self.repair_procedure = repair_procedure
        self.repair_is_neural = isinstance(self.repair_procedure, NeuralProcedure)

    def _evaluate(self, data: DataLoader, batch_size: int) -> Dict:
        """
        Perform evaluation on data.

        Args:
            data (DataLoader): Input data to be evaluated.
            batch_size (int): Batch size dimension for evaluation.

        Returns:
            Dict: Evaluation results with keys:
                    "target_solution": Mean score of solutions found by the solver
                    "incumbent_solution": Mean score of solutions found by the neural operator
                    "time": Time taken to compute the solution
        """
        target_cost = list()
        incumbent_cost = list()
        for batch_idx, batch in tqdm(enumerate(chunked(data, batch_size)),
                                         desc="Validation batch",
                                         leave=False,
                                         total=len(data) // batch_size):
            # work on copies
            batch = [deepcopy(s) for s in batch]

            target_cost.extend((s.cost for s in batch))

            start_time = time.time()
            with torch.no_grad():
                self.destroy_procedure.multiple(batch)
                self.repair_procedure.multiple(batch)
            runtime = time.time() - start_time

            incumbent_cost.extend((sol.cost for sol in batch))

        return {"target_solution": np.mean(target_cost),
                "incumbent_solution": np.mean(incumbent_cost),
                "time": runtime }

    def train(self,
              train: DataLoader,
              epochs: int = 1,
              batch_size: int = 1,
              validation: DataLoader = None,
              val_interval: int = None,
              val_batch_size: int = 1,
              log_interval: int = None,
              logger: Logger = EmptyLogger()):
        """
        Train the neural procedur pair by both destroy and repair, when needed.

        WIP: currently subject to changes to make the repair neural
        operator working again. In order to train the destroy operator,
        see train_destroy.py.

        Args:
            train (DataLoader): Training data.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 1.
            validation (DataLoader, optional): Validation data. Defaults to None.
            val_interval (int, optional): number of steps between validation runs. Defaults to None.
            val_batch_size (int, optional): Validation batch size. Defaults to 1.
            log_interval (int, optional): number of steps between logging messages. Defaults to None.
            logger (Logger, optional): Logger. Defaults to EmptyLogger().
        """
        logger.new_run(f'{type(self.destroy_procedure).__name__}-'
                       f'{type(self.repair_procedure).__name__}')

        can_evaluate = validation is not None and val_interval is not None
        start_time = time.time()

        if self.destroy_is_neural:
            self.destroy_procedure._init_train()
        if self.repair_is_neural:
            self.repair_procedure._init_train()

        # Pregenerate initial solutions
        initial_solutions = [
            VRPNeuralSolution.from_solution(
                nearest_neighbor_solution(inst))
            for inst in train]

        for epoch in tqdm(range(epochs), desc="Training epoch"):
            for batch_idx, batch in tqdm(
                enumerate(chunked(initial_solutions, batch_size)),
                desc="Training batch",
                leave=False,
                total=len(initial_solutions) // batch_size):

                # batch = [deepcopy(s) for s in batch]
                batch_costs = [s.cost for s in batch]
                mean_batch_cost = sum(batch_costs) / len(batch_costs)

                # Training depends on the procedures configuration:
                # if we are training a destroy procedure and the repair is not
                # neural then we only need to take that into account
                # if we are training both a destroy
                # procedure and a repair procedure then we need to take
                # care of the fact that the training procedure is performed
                # using Reinforcement Learning and hence the global loss
                # is the one obtained from the repair operator.
                if self.destroy_is_neural:
                    pred, loss, info = self.destroy_procedure._train_step(batch)
                else:
                    self.destroy_procedure(batch)
                    pred = batch

                if self.repair_is_neural:
                    # turn batch back into PyG format as destroy procedure
                    # outputs a list of VRPSolution
                    self.repair_procedure._train_step(pred)
                else:
                    pred = self.repair_procedure.multiple(pred)

                repaired_costs = [p.cost for p in pred]
                mean_repaired_cost = sum(repaired_costs) / len(repaired_costs)

                if can_evaluate and batch_idx % val_interval == 0:
                    evaluation_results = self._evaluate(validation, batch_size=val_batch_size)
                    evaluation_results["epoch"] = epoch
                    evaluation_results["epoch_step"] = batch_idx
                    logger.log(evaluation_results, "validation")

                if log_interval is not None and batch_idx % log_interval == 0:
                    log_dict = {
                        'target_solution': mean_batch_cost,
                        'incumbent_solution': mean_repaired_cost
                    }

                    if self.destroy_is_neural:
                        log_dict.update(
                            self.destroy_procedure._train_info(
                                epoch, batch_idx, log_interval))

                    if self.repair_is_neural:
                        log_dict.update(
                            self.repair_procedure._train_info(
                                epoch, batch_idx, log_interval))

                    logger.log(log_dict, 'train')
                    # logger.log({
                    #     "target_solution": mean_batch_cost,
                    #     "incumbent_solution": mean_repaired_cost,
                    #     "time": "NA",
                    #     "epoch": epoch,
                    #     "epoch_step": batch_idx,
                    # }, "training")

        print(f"Training completed successfully in {time.time() - start_time} seconds.")