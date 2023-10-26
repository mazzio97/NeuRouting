import argparse
import math

import os
from glob import glob

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import tqdm

from nlns.generators.dataset import NazariDataset

def n_customers(s):
  if "," not in s:
    return int(s)
  else:
    try:
      m, M = map(int, s.split(','))
      return m, M
    except:
      raise argparse.ArgumentTypeError("Number of customers must be in format <min>,<max> if randomly sampled num of customers is used.")

parser = argparse.ArgumentParser(description="Generare dataset based on Nazari")
parser.add_argument('-i', '--instances', type=int, required=True)
parser.add_argument('-n', '--n_customers', type=n_customers, required=True)
parser.add_argument('-o', '--out', type=str, required=True)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('--lkh-runs', type=int, required=False, default=1)

args = parser.parse_args()

if __name__ == "__main__":
  pl.seed_everything(args.seed)
  dataset = NazariDataset(args.instances, args.n_customers,
                          solve=True,
                          lkh_runs=args.lkh_runs,
                          save_path=args.out)
  
  dl = DataLoader(dataset,
                  batch_size=1,
                  shuffle=False,
                  num_workers=os.cpu_count(), 
                  prefetch_factor=5, 
                  collate_fn=lambda x: x)

  iter_data = tqdm.tqdm(enumerate(iter(dl)), total=args.instances)
  file_size = -1
  for idx, _ in iter_data:
    if idx == 0:
      size = os.path.getsize(glob(os.path.join(args.out, "*.vrp"))[0])
      size += os.path.getsize(glob(os.path.join(args.out, "*.sol"))[0])
      file_size = size / 1e6
    iter_data.set_description_str(f"{file_size * (idx + 1):5.2} MB", refresh=True)
      
