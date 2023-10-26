import sys
sys.path.append("src")


from more_itertools import chunked
from tqdm.auto import tqdm
import numpy as np
import torch
import argparse
from utils.logging import MultipleLogger, TabularConsoleLogger, WandBLogger
from nlns.destroy import ResGatedGCNDestroy
from generators.dataset import NazariDataset
from time import time


parser = argparse.ArgumentParser(description='Train neural destroy operatos')
parser.add_argument('-o', '--out', type=str, required=True)
parser.add_argument('-n', '--n_customers', type=int, required=True)
parser.add_argument('-s', '--steps', type=int, required=True)
parser.add_argument('-v', '--validation', type=int, default=100)
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-vb', '--validation_batch_size', type=int, default=256)
parser.add_argument('-vi', '--validation_interval', type=int, required=False)
parser.add_argument('-log', '--log_interval', type=int, required=False)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('-p', '--patience', type=int, required=False, default=5)

args = parser.parse_args()

if __name__ == "__main__":
    seed = args.seed
    steps = args.steps
    batch_size = args.batch_size
    n_customers = args.n_customers
    log_interval = args.log_interval

    save_path = args.out
    patience = args.patience

    val_samples = args.validation
    val_batch_size = args.validation_batch_size
    val_interval = args.validation_interval

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train = NazariDataset(steps * batch_size, n_customers,
                          batch_size=steps * batch_size,
                          lkh_pass=10,
                          lkh_runs=1,
                          workers=batch_size)
    # validation dataset is consolidated into a list to always use the same set
    validation = list(NazariDataset(val_samples, n_customers,
                      batch_size=batch_size, lkh_pass=None, lkh_runs=1, workers=val_batch_size))

    logger = TabularConsoleLogger(["phase", "step", "loss"])

    destroy = ResGatedGCNDestroy(0, model_config={"device": device})
    destroy._init_train()

    start_time = time()
    batch_loss = list()
    last_validation_loss = 100
    patience_strikes = 0
    for step_idx, batch in tqdm(enumerate(chunked(train, batch_size)),
                                 desc="Training step",
                                 total=len(train) // batch_size):
        # stop training if patience is reached
        if patience_strikes > patience: break
        pred, loss, info = destroy._train_step(batch)
        batch_loss.append(loss)

        if log_interval is not None and step_idx % log_interval == 0:
            logger.log(
                { "step": step_idx, "loss": np.mean(batch_loss) },
                "training")
            batch_loss = list()

        if step_idx % val_interval == 0:
            val_losses = list()
            for val_batch_idx, val in tqdm(enumerate(chunked(validation, val_batch_size)),
                                           desc="Validation batch",
                                           leave=False,
                                           total=len(validation) // val_batch_size):
                with torch.no_grad():
                    pred, loss, info = destroy._evaluate(val)
                    val_losses.append(loss)

            val_loss = np.mean(val_losses)
            patience_strikes = 0 if val_loss < last_validation_loss else patience_strikes + 1
            if patience_strikes > patience:
                # stop training since maximum patience has been reached
                break

            last_validation_loss = val_loss
            logger.log({ "steps": step_idx, "loss": val_loss }, "validation")
    end_time = time()

    print(f"Training ended in {(end_time - start_time)}s")
    print(f"Saving model to {save_path}")
    destroy.save(save_path)
