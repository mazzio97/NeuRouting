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

from tqdm.auto import tqdm

from generators.dataset import instance_to_PyG
from environments.batch_lns_env import BatchLNSEnvironment
from instances import VRPInstance, VRPSolution, VRPNeuralSolution
from nlns import LNSProcedure, RepairProcedure, DestroyProcedure, LNSOperator
from nlns.initial import nearest_neighbor_solution
from utils.logging import Logger, EmptyLogger

class NeuralProcedure(LNSProcedure):
    @abstractmethod
    def _init_train(self):
        """
        Initialize training procedure by instantiating the optimizer
        and putting the model in training mode.
        """
        pass

    @abstractmethod
    def _train_step(self, data: List[VRPSolution]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Perform a training step on the input data.

        Args:
            data (List[VRPSolution]): Input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Tuple in the form (out, loss, training_info).
                training_info is a dict where the keys are the information names
                and the values the information values. out is the partial solution on which
                the model has been applied.
        """
        pass

    @abstractmethod
    def _evaluate(self, data: List[VRPSolution]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Perform evaluation on the input data.

        Args:
            data (List[VRPSolution]): Input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Tuple in the form (out, loss, training_info).
                training_info is a dict where the keys are the information names
                and the values the information values. out is the partial solution on which
                the model has been applied.
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict:
        """
        Returns:
            Dict: State dict containing all relevant informations for a Neural procedure to be saved.
        """
        pass

    def save(self, path: str):
        """
        Save the trained neural procedure to path.

        Args:
            path (str): Output path
        """
        torch.save(self.state_dict(), path)

    @abstractclassmethod
    def load(cls, path: str, device: str = "cpu"):
        """
        Load model saved in path.

        Args:
            path (str): Path of saved model
            device (str): Device on which the model is loaded
        """
        pass


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
        can_evaluate = validation is not None and val_interval is not None
        start_time = time.time()

        if self.destroy_is_neural:
            self.destroy_procedure._init_train()
        if self.repair_is_neural:
            self.repair_procedure._init_train()

        for epoch in tqdm(range(epochs), desc="Training epoch"):
            for batch_idx, batch in tqdm(enumerate(chunked(train, batch_size)), 
                                         desc="Training batch", 
                                         leave=False,
                                         total=len(train) // batch_size):
                # work on copies
                batch = [deepcopy(s) for s in batch]
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
                    pred = self.destroy_procedure.multiple(batch)

                if self.repair_is_neural:
                    # turn batch back into PyG format as destroy procedure 
                    # outputs a list of VRPSolution
                    pred, loss, info = self.repair_procedure._train_step(pred)
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
                    logger.log({
                        "target_solution": mean_batch_cost,
                        "incumbent_solution": mean_repaired_cost,
                        "time": "NA",
                        "epoch": epoch,
                        "epoch_step": batch_idx,
                    }, "training")

        print(f"Training completed successfully in {time.time() - start_time} seconds.")