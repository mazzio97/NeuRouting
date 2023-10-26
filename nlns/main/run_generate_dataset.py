import argparse
import sys
import os
import itertools
from more_itertools import chunked
import time

sys.path.append("src")

from tqdm.auto import tqdm
from generators.dataset import NazariDataset

#parser = argparse.ArgumentParser(description='Generate VRP instances')
#parser.add_argument('-i', '--n_instances', type=int, required=True)
#parser.add_argument('-c', '--n_customers', nargs='+', type=int, required=True)
#parser.add_argument('-o', '--out', type=str, required=True)
#parser.add_argument('-d', '--distribution', type=str, required=False, default='nazari')

#args = parser.parse_args()

import time

if __name__ == "__main__":
    #logger = ConsoleLogger()
    
    workers = 5
    time_s = time.time()
    list(NazariDataset(5, (50, 100), solve=True, workers=workers, lkh_runs=1, lkh_pass=1))
    time_e = time.time()
    print(f"{workers} worker: {time_e - time_s}")