import abc
import time
import contextlib
import warnings
from math import ceil
from typing import Dict, Optional, Union
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import DataLoader
from more_itertools import chunked
from tqdm.auto import tqdm

from nlns.models import RLAgentSolution
from nlns.operators.initial import nearest_neighbor_solution
from nlns.operators import LNSOperator
from nlns.utils.logging import Logger, EmptyLogger


class Trainable(abc.ABC):
    """Interface for trainable operators.

    Implementing this in an operator is necessary in order for said
    operator to be used with :class:`NLNSTrainer`. This is particularly
    useful for operators whose train procedure is highly correlated to
    the large neighborhood search environment.
    """

    def init_train(self):
        """Initialize training (e.g. optimizers, statistics, etc)."""

    @abc.abstractmethod
    def training_step(self, batch):
        """Train step on a given batch."""

    @abc.abstractmethod
    def training_info(self, epoch, batch_idx,
                      log_interval) -> Dict[str, float]:
        """Return train statistics (e.g. loss, reward, etc.)."""

    def save(self, path, epoc, batch_idx):
        """Save current checkpoint."""
        warnings.warn('Saving was not implemented for this operator '
                      f'({self.__name__})')


class NLNSTrainer:
    """"""

    def __init__(self, destroy_operator: Union[Trainable, LNSOperator],
                 repair_operator: Union[Trainable, LNSOperator]):
        self.destroy_trainable = isinstance(destroy_operator, Trainable)
        self.repair_trainable = isinstance(repair_operator, Trainable)

        if not self.repair_trainable and not self.destroy_trainable:
            raise TypeError('At least one of the given operators must '
                            'be trainable (implement Trainable). '
                            f'{destroy_operator} and {repair_operator} '
                            'are not.')

        self.destroy_operator = destroy_operator
        self.repair_operator = repair_operator

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
        # TODO: set operators into valid mode for evaluation.
        # If we don't want to assume that `model` is a torch module inside
        # the operator, maybe add train() and eval() methods to the Trainable
        # interface.
        start_time = time.time()

        validation_solutions = [
            RLAgentSolution.from_solution(
                nearest_neighbor_solution(instance)) for
            instance in data]
        costs = [solution.cost for solution in validation_solutions]

        # Assuming all the instances have the same number of customers
        for _ in range(validation_solutions[0].instance.n_customers):
            backup_copies = [deepcopy(sol) for sol in validation_solutions]
            n_solutions = len(validation_solutions)

            n_batches = ceil(n_solutions / batch_size)
            for i in range(n_batches):
                begin = i * batch_size
                end = min((i + 1) * batch_size, n_solutions)
                self.destroy_operator(validation_solutions[begin:end])
                self.repair_operator(validation_solutions[begin:end])

            for i in range(n_solutions):
                cost = validation_solutions[i].cost
                # Only "accept" improving solutions
                if costs[i] < cost:
                    validation_solutions[i] = backup_copies[i]
                else:
                    costs[i] = cost

        return {"mean_cost": np.mean(costs),
                "time": time.time() - start_time}

    def init_train(self):
        """Initialize training of given operators, if trainable."""
        if self.destroy_trainable:
            self.destroy_operator.init_train()

        if self.repair_trainable:
            self.repair_operator.init_train()

    def training_step(self, batch):
        """Compute a train step for the operators, if trainable."""
        if self.destroy_trainable:
            self.destroy_operator.training_step(batch)
        else:
            self.destroy_operator(batch)

        if self.repair_trainable:
            self.repair_operator.training_step(batch)
        else:
            self.repair_operator(batch)

    def training_info(self, epoch, batch_idx, log_interval) -> Dict:
        """Retrieve train info from given operators.

        Returns:
            A dict in the form::

                {'destroy/param1': {...}, 'destroy/param2': {...},
                 'repair/param1': {...}, ...}

            Each key will be present only if the corresponding operator
            is trainable.
        """
        info = {}

        if self.destroy_trainable:
            destroy_info = self.destroy_operator.training_info(
                epoch, batch_idx, log_interval)

            for key, value in destroy_info.items():
                info[f'destroy/{key}'] = value

        if self.repair_trainable:
            repair_info = self.repair_operator.training_info(
                epoch, batch_idx, log_interval)

            for key, value in repair_info.items():
                info[f'repair/{key}'] = value

        return info

    def train(self,
              train: DataLoader,
              epochs: int = 1,
              batch_size: int = 1,
              validation: DataLoader = None,
              val_interval: int = None,
              val_batch_size: int = 1,
              log_interval: int = None,
              initial_solution_fn=nearest_neighbor_solution,
              logger: Logger = EmptyLogger()):
        """Train the operators in a LNS-like environment.

        This mimics the work of `[Huttong & Tierney] <https://doi.org/10.48550/arXiv.1911.09539>`_.

        TODO: Add verbosity options

        Args:
            train (DataLoader): Training data.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 1.
            validation (DataLoader, optional): Validation data. Defaults to None.
            val_interval (int, optional): number of steps between validation runs. Defaults to None.
            val_batch_size (int, optional): Validation batch size. Defaults to 1.
            log_interval (int, optional): number of steps between logging messages. Defaults to None.
            logger (Logger, optional): Logger. Defaults to EmptyLogger().
        """                 # NOQA
        logger.new_run(f'{type(self.destroy_operator).__name__}-'
                       f'{type(self.repair_operator).__name__}')

        can_evaluate = validation is not None and val_interval is not None
        start_time = time.time()

        self.init_train()

        # Pregenerate initial solutions
        initial_solutions = [initial_solution_fn(inst)
                             for inst in train]

        for epoch in tqdm(range(epochs), desc='Training epoch'):
            for batch_idx, batch in tqdm(
                enumerate(chunked(initial_solutions, batch_size)),
                desc='Training batch',
                leave=False,
                    total=len(initial_solutions) // batch_size):

                # batch = [deepcopy(s) for s in batch]
                batch_costs = [s.cost for s in batch]
                mean_batch_cost = sum(batch_costs) / len(batch_costs)

                # Batch is updated inline
                self.training_step(batch)

                repaired_costs = [p.cost for p in batch]
                mean_repaired_cost = sum(repaired_costs) / len(repaired_costs)

                if can_evaluate and batch_idx % val_interval == 0:
                    evaluation_results = self._evaluate(
                        validation,
                        batch_size=val_batch_size)
                    evaluation_results['epoch'] = epoch
                    evaluation_results['epoch_step'] = batch_idx
                    logger.log(evaluation_results, 'validation')

                if log_interval is not None and batch_idx % log_interval == 0:
                    log_dict = {
                        'target_solution': mean_batch_cost,
                        'incumbent_solution': mean_repaired_cost
                    }

                    log_dict.update(self.training_info(epoch, batch_idx,
                                                       log_interval))

                    logger.log(log_dict, 'train')

        print('Training completed successfully in '
              f'{time.time() - start_time} seconds.')


class TorchReproducibilityMixin:
    """Provide reproducibility for torch based operators.

    Mixin designed to be used with :class:`nlns.LNSOperator` subclasses.
    """
    _torch_reproducibility: bool = False
    torch_rng_state: Optional[torch.Tensor] = None

    def init_torch_reproducibility(self, seed: Optional[int]):
        """Enable reproducible results for the torch operator.

        To be used inside
        :meth:`nlns.operators.LNSOperator.set_random_state`.
        """
        self._torch_reproducibility = True
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.torch_rng_state = torch.get_rng_state()

    @property
    def torch_reproducibility(self) -> bool:
        """Return whether reproducibility for the operator is enabled."""
        return self._torch_reproducibility

    @contextlib.contextmanager
    def sync_torch_rng_state(self):
        """Context manager: fork random state and reproducibility.

        To be used at inference time to correctly fork rng state without
        contaminating global state.
        """
        # Do nothing if reproducibility is not active
        if not self._torch_reproducibility:
            yield
            return

        with torch.random.fork_rng() as fork:
            torch.set_rng_state(self.torch_rng_state)
            torch.use_deterministic_algorithms(True)
            try:
                yield [fork]
            finally:
                self.torch_rng_state = torch.get_rng_state()

            torch.use_deterministic_algorithms(False)
