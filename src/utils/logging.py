from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List

import wandb
from tqdm.auto import tqdm

from tabulate import tabulate

class Logger(ABC):
    @abstractmethod
    def new_run(self, run_name: str):
        pass

    @abstractmethod
    def log(self, info: Dict[str, Number], phase: str):
        pass

class EmptyLogger(Logger):
    def new_run(self, run_name):
        pass

    def log(self, info, phase):
        pass

class WandBLogger(Logger):
    def __init__(self, project: str = "NeuRouting", username: str = "mazzio97"):
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


class TabularConsoleLogger(Logger):
    def __init__(self, headers: List = []):
        self._content = { k: [] for k in headers }
        self._headers_printed = False

    def new_run(self, run_name: str):
        pass

    def log(self, info: Dict[str, Number], phase: str):
        for k in self._content.keys():
            v = info.get(k, "") if k != "phase" else phase
            self._content[k].append(v)

        s = tabulate(self._content, headers="keys")
        
        if self._headers_printed:
            s = s.split("\n")[-2]
        else:
            self._headers_printed = True
        
        tqdm.write(s)


class ConsoleLogger(Logger):
    def __init__(self):
        self._header_printed = False

    def new_run(self, run_name: str):
        pass

    def log(self, info: Dict[str, Number], phase: str):
        content = { k:[v] for k, v in info.items() }
        content["phase"] = [phase]
        
        s = tabulate(content, headers="keys")
        
        if self._header_printed:
            s = s.split("\n")[2]
        else:
            self._header_printed = True
        
        tqdm.write(s)


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
