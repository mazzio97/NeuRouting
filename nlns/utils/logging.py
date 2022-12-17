from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List

import wandb


class Logger(ABC):
    @abstractmethod
    def new_run(self, run_name: str):
        pass

    @abstractmethod
    def log(self, info: Dict[str, Number], phase: str):
        pass


class EmptyLogger(Logger):
    """Drop all logs, to be used as default logger."""

    def new_run(self, run_name: str):
        pass

    def log(self, info: Dict[str, Number], phase: str):
        pass


class WandBLogger(Logger):
    def __init__(self, username: str, project: str = "NeuRouting"):
        # self.model = model
        self.project = project
        self.username = username

    def new_run(self, run_name=None):
        wandb.init(name=run_name, project=self.project, entity=self.username)
        # wandb.watch(self.model)

    def log(self, info: Dict[str, Number], phase: str):
        keys = []
        for k in info.keys():
            keys.append(f"{phase}/{k}")
        info = dict(zip(keys, list(info.values())))
        wandb.log(info)


class ConsoleLogger(Logger):
    def new_run(self, run_name: str):
        pass

    def log(self, info: Dict[str, Number], phase: str):
        print(f"[{phase.upper()}]")
        for k, v in info.items():
            print(f"{k}: {v}")


class MultipleLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = set() if loggers is None else set(loggers)

    def add(self, logger: Logger):
        self.loggers.add(logger)

    def remove(self, logger: Logger):
        self.loggers.remove(logger)

    def new_run(self, run_name: str):
        for logger in self.loggers:
            logger.new_run(run_name)

    def log(self, info: Dict[str, Number], phase: str):
        for logger in self.loggers:
            logger.log(info, phase)

