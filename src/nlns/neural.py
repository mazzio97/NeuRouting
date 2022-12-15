import ntpath
import os
import time
from abc import abstractmethod
from math import ceil
from typing import Union, Optional, List, Callable

import numpy as np
import torch
from torch import nn

from environments.batch_lns_env import BatchLNSEnvironment
from instances import VRPInstance, VRPNeuralSolution
from nlns import LNSProcedure, RepairProcedure, DestroyProcedure, LNSOperator
from nlns.initial import nearest_neighbor_solution
from utils.logging import Logger


class NeuralProcedure(LNSProcedure):
    def __init__(self, model: nn.Module, device: str = "cpu", logger: Optional[Logger] = None):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.val_env = None
        self._val_phase = False

    def train(self,
              train_instances: List[VRPInstance],
              opposite: Union[DestroyProcedure, RepairProcedure],
              n_epochs: int,
              batch_size: int,
              ckpt_path: str,
              initial: Callable = nearest_neighbor_solution,
              val_instances: Optional[List[VRPInstance]] = None,
              log_interval: Optional[int] = None,
              val_interval: Optional[int] = None):

        run_name = ntpath.basename(ckpt_path)
        run_name = run_name[:run_name.rfind('.')]
        if self.logger is not None:
            self.logger.new_run(run_name=run_name)

        incumbent_cost = np.inf
        train_size = len(train_instances)
        n_batches = ceil(float(train_size) / batch_size)
        if val_instances is not None:
            n_customers = val_instances[0].n_customers
            val_steps = n_customers
            log_interval = log_interval if log_interval is not None else n_batches
            val_interval = val_interval if val_interval is not None else n_batches

        start_time = time.time()
        print(f"Starting {run_name} training...")
        train_solutions = [initial(inst) for inst in train_instances]

        if isinstance(self, RepairProcedure):
            train_solutions = [VRPNeuralSolution.from_solution(sol) for sol in train_solutions]

        self._init_train()
        for epoch in range(n_epochs):
            for batch_idx in range(n_batches):
                begin = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, train_size)
                self.model.train()

                self._train_step(opposite, train_solutions[begin:end])

                if self.logger is not None and (batch_idx + 1) % log_interval == 0:
                    self.logger.log(self._train_info(epoch, batch_idx, log_interval), phase="train")

                if val_instances is not None and ((batch_idx + 1) % val_interval == 0 or batch_idx + 1 == n_batches):
                    self.model.eval()
                    start_eval_time = time.time()
                    solutions = self.validate(val_instances, batch_size, val_steps, opposite, initial)
                    runtime = time.time() - start_eval_time
                    mean_cost = np.mean([sol.cost() for sol in solutions])
                    if self.logger is not None:
                        self.logger.log({"epoch": epoch + 1,
                                         "batch_idx": batch_idx + 1,
                                         "mean_cost": mean_cost,
                                         "runtime": runtime}, phase="val")
                    if mean_cost < incumbent_cost:
                        incumbent_cost = mean_cost
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        torch.save(self._ckpt_info(epoch, batch_idx), ckpt_path)
        self.val_env = None
        print(f"Training completed successfully in {time.time() - start_time} seconds.")

    def validate(self,
                 instances: List[VRPInstance],
                 batch_size: int,
                 steps: int,
                 opposite_procedure: Union[DestroyProcedure, RepairProcedure],
                 initial: Callable):
        # Fix the seed to reduce the sources of randomness while evaluating the model
        np.random.seed(42)
        torch.manual_seed(42)
        self._val_phase = True
        if self.val_env is None:
            if isinstance(opposite_procedure, RepairProcedure) and isinstance(self, DestroyProcedure):
                self.val_env = BatchLNSEnvironment(batch_size, LNSOperator(self, opposite_procedure), initial)
            elif isinstance(opposite_procedure, DestroyProcedure) and isinstance(self, RepairProcedure):
                self.val_env = BatchLNSEnvironment(batch_size, LNSOperator(opposite_procedure, self), initial)
            else:
                self.val_env = None
            assert self.val_env is not None, f"{opposite_procedure} and {self} should be two opposite LNS procedures."
        solutions = self.val_env.solve(instances, max_steps=steps)
        self._val_phase = False
        return solutions

    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, self.device)
        self.model.load_state_dict(ckpt["parameters"])
        self.model.eval()

    @abstractmethod
    def _init_train(self):
        pass

    @abstractmethod
    def _train_step(self, opposite_procedure, train_batch):
        pass

    @abstractmethod
    def _train_info(self, epoch, batch_idx, log_interval) -> dict:
        pass

    @abstractmethod
    def _ckpt_info(self, epoch, batch_idx) -> dict:
        pass
