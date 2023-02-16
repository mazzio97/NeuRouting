import argparse
import math

import pytorch_lightning as pl
import torch

from nlns.models.res_gated_gcn import ResGatedGCN
from nlns.models.dataloader import DataModule

def n_customers(s):
  if "," not in s:
    return int(s)
  else:
    try:
      m, M = map(int, s.split(','))
      return m, M
    except:
      raise argparse.ArgumentTypeError("Number of customers must be in format <min>,<max> if randomly sampled num of customers is used.")

parser = argparse.ArgumentParser(description='Train neural destroy operator')
parser.add_argument('-o', '--out', type=str, required=True)
parser.add_argument('-n', '--n_customers', type=n_customers, required=True)
parser.add_argument('-v', '--valid', type=int, required=True)
parser.add_argument('-b', '--batch-size', type=int, required=False, default=16)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('--distribution', type=str, required=False, default="nazari")
parser.add_argument('--log-interval', type=int, required=False, default=100)
parser.add_argument('--num-neighbors', type=int, required=False, default=20)
parser.add_argument('--steps-per-epoch', type=int, required=False, default=500)
parser.add_argument('--max-epochs', type=int, required=False, default=1500)
parser.add_argument('--initial-lr', type=float, required=False, default=0.001)
parser.add_argument('--wandb-name', type=str, required=False, default=None)
parser.add_argument('--save', type=str, required=False, default=None)

args = parser.parse_args()

if __name__ == "__main__":
  # following Joshi (2019) and Kool (2022) we train for at most 1500
  # epochs with 500 steps for each epoch.
  data = DataModule(num_nodes=args.n_customers,
                    valid_instances=args.valid,
                    steps_per_epoch=args.steps_per_epoch,
                    num_neighbors=args.num_neighbors,
                    batch_size=args.batch_size,
                    save_path=args.save)
  
  destroy = ResGatedGCN(num_neighbors=args.num_neighbors, 
                        initial_learning_rate=args.initial_lr)
  wandb_logger = pl.loggers.WandbLogger(project="NeuRouting", name=args.wandb_name)
 
  trainer = pl.Trainer(max_epochs=args.max_epochs,
                       devices=1,
                       accelerator="auto",
                       logger=wandb_logger,
                       log_every_n_steps=args.log_interval,
                       check_val_every_n_epoch=5,
                       callbacks=[
                         pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
                         pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="valid/loss", mode="min", 
                                                      dirpath=args.out, filename="destroy-{epoch}",
                                                      every_n_epochs=5)
                       ])

  trainer.fit(destroy, datamodule=data)