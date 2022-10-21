from typing import Optional

from generators import generate_multiple_instances
from nlns.builder import destroy_procedures, repair_procedures, neural_models, get_neural_procedure
from models import VRPCriticModel
from utils.logging import Logger


class Trainer:
    def __init__(self,
                 n_customers: int,
                 n_train_instances: int,
                 n_val_instances: int,
                 distribution: str = "nazari",
                 device: str = "cpu",
                 ckpt_path: str = "./pretrained/",
                 logger: Optional[Logger] = None):

        self.n_customers = n_customers
        self.train_instances = generate_multiple_instances(n_instances=n_train_instances,
                                                           n_customers=n_customers,
                                                           distribution=distribution,
                                                           seed=42)
        self.val_instances = generate_multiple_instances(n_instances=n_val_instances,
                                                         n_customers=n_customers,
                                                         distribution=distribution,
                                                         seed=73)
        self.device = device
        self.ckpt_path = ckpt_path
        self.logger = logger

    def train(self, model_name: str, opposite_name: str, epochs: int, batch_size: int, destroy_percentage: float,
                        log_interval: Optional[int] = None, val_interval: Optional[int] = None):
        neural_proc, model, _ = get_neural_procedure(model_name, opposite_name, destroy_percentage, self.ckpt_path)
        if model_name in destroy_procedures.keys():
            neural_proc = neural_proc(model, destroy_percentage, device=self.device, logger=self.logger)
        else:
            neural_proc = neural_proc(model, VRPCriticModel(), device=self.device, logger=self.logger)

        if opposite_name in neural_models.keys():
            proc, model, ckpt = get_neural_procedure(opposite_name, model_name, destroy_percentage, self.ckpt_path)
            opposite_proc = proc(model, device=self.device) if opposite_name in repair_procedures.keys() else \
                proc(model, destroy_percentage, device=self.device)
            if ckpt is not None:
                print(f"Loading {ckpt} destroy checkpoint...")
                opposite_proc.load_model(ckpt)
        elif opposite_name in destroy_procedures.keys():
            opposite_proc = destroy_procedures[opposite_name](destroy_percentage)
        else:
            opposite_proc = repair_procedures[opposite_name]()

        ckpt_file = f"model_{model_name}_opposite_{opposite_name}_n{self.n_customers}_p{destroy_percentage}.pt"

        return neural_proc.train(train_instances=self.train_instances,
                                 val_instances=self.val_instances,
                                 opposite=opposite_proc,
                                 n_epochs=epochs,
                                 batch_size=batch_size,
                                 ckpt_path=self.ckpt_path + ckpt_file,
                                 log_interval=log_interval,
                                 val_interval=val_interval)
