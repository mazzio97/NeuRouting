import sys
import torch
import pytorch_lightning as pl

sys.path.append("src")

#from main.trainer import Trainer
#from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger
from models.res_gated_gcn import ResGatedGCN
#from nlns.repair import GreedyRepair
#from nlns.neural import NeuralProcedurePair

from models.dataloader import DataModule


if __name__ == "__main__":
    num_neighbors = 20


    data = DataModule(num_nodes=20,
                      train_instances=100,
                      valid_instances=2,
                      batch_size=4,
                      num_neighbors=num_neighbors)    
    
    destroy = ResGatedGCN(num_neighbors=num_neighbors)
    
    wandb_logger = pl.loggers.WandbLogger(project="NeuRouting")
    trainer = pl.Trainer(max_epochs=1,
                         logger=wandb_logger)
    trainer.fit(destroy, datamodule=data)
    