import time
import contextlib
from math import ceil
from typing import Dict, Optional
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import DataLoader
from more_itertools import chunked
from tqdm.auto import tqdm

from nlns.models import RLAgentSolution
from nlns.operators.initial import nearest_neighbor_solution
from nlns.utils.logging import Logger, EmptyLogger


# class NeuralProcedure:
#     def __init__(self, model: nn.Module, device: str = "cpu", logger: Optional[Logger] = None):
#     self.model = model.to(device)
#     self.device = device
#     self.logger = logger
#     self.val_env = None
#     self._val_phase = False


class NeuralProcedurePair:
    """
    A neural procedure pair is used to train one (or two) neural procedure(s).
    This is done by providing a repair and a destroy procedure.
    Either one or the other procedures needs to be neural based.
    """
    def __init__(self, destroy_procedure, repair_procedure):
        self.destroy_procedure = destroy_procedure
        self.destroy_is_neural = False
        self.repair_procedure = repair_procedure
        self.repair_is_neural = True

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
                self.destroy_procedure(validation_solutions[begin:end])
                self.repair_procedure(validation_solutions[begin:end])

            for i in range(n_solutions):
                cost = validation_solutions[i].cost
                # Only "accept" improving solutions
                if costs[i] < cost:
                    validation_solutions[i] = backup_copies[i]
                else:
                    costs[i] = cost

        return {"mean_cost": np.mean(costs),
                "time": time.time() - start_time}

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
            RLAgentSolution.from_solution(
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
                    pred = self.repair_procedure(pred)

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
