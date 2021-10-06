import os
import random
from typing import List, Dict, Union

import numpy as np

from environments.lns_env import LNSEnvironment
from environments.sa_lns_env import SimAnnealingLNSEnvironment

from nlns import LNSOperator
from nlns.initial import nearest_neighbor_solution
from nlns.destroy import DestroyRandom, DestroyPointBased, DestroyTourBased, ResidualGatedGCNDestroy
from nlns.repair import GreedyRepair, SCIPRepair, RLAgentRepair

from models import ResidualGatedGCNModel, VRPActorModel


destroy_procedures = {
    "random": DestroyRandom,
    "point": DestroyPointBased,
    "tour": DestroyTourBased,
    "resgatedgcn": ResidualGatedGCNDestroy,
}

repair_procedures = {
    "greedy": GreedyRepair,
    "scip": SCIPRepair,
    "rlagent": RLAgentRepair,
}

neural_models = {
    "resgatedgcn": ResidualGatedGCNModel(),
    "rlagent": VRPActorModel(),
}


def nlns_builder(destroy_names: Dict[str, Union[float, List[float]]],
                 repair_names: List[str],
                 neighborhood_size: int,
                 initial=nearest_neighbor_solution,
                 simulated_annealing=False,
                 name="nlns",
                 device="cpu",
                 ckpt_path="./pretrained/") -> LNSEnvironment:
    lns_operators_pairs = np.array(np.meshgrid(list(destroy_names.keys()), repair_names)).T.reshape(-1, 2)

    lns_operators = []
    for destroy, repair in lns_operators_pairs:
        destroy_p = destroy_names[destroy]
        if type(destroy_p) is float:
            destroy_p = [destroy_p]
        for percentage in destroy_p:
            lns_operators.append(get_lns_operator(destroy, repair, percentage, device, ckpt_path))

    lns_env = SimAnnealingLNSEnvironment if simulated_annealing else LNSEnvironment

    return lns_env(lns_operators, neighborhood_size, initial, name=name)


def get_lns_operator(destroy_name: str,
                     repair_name: str,
                     destroy_percentage: float,
                     device="cpu",
                     ckpt_path="./pretrained/") -> LNSOperator:
    if destroy_name in neural_models.keys():
        proc, model, ckpt = get_neural_procedure(destroy_name, repair_name, destroy_percentage, ckpt_path)
        destroy = proc(model, destroy_percentage, device=device)
        if ckpt is not None:
            print(f"Loading {ckpt} checkpoint...")
            destroy.load_model(ckpt)
    else:
        destroy = destroy_procedures[destroy_name](destroy_percentage)

    if repair_name in neural_models.keys():
        proc, model, ckpt = get_neural_procedure(repair_name, destroy_name, destroy_percentage, ckpt_path)
        repair = proc(model, device=device)
        if ckpt is not None:
            print(f"Loading {ckpt} checkpoint...")
            repair.load_model(ckpt)
    else:
        repair = repair_procedures[repair_name]()
    return LNSOperator(destroy, repair)


def get_neural_procedure(neural_name: str, opposite_name: str, percentage: float, ckpt_path: str):
    assert neural_name in neural_models.keys(), \
        f"Unknown neural procedure {neural_name}, select one between {neural_models.keys()}."

    neural_proc = repair_procedures[neural_name] if opposite_name in destroy_procedures.keys() \
        else destroy_procedures[neural_name]
    neural_model = neural_models[neural_name]
    model_ckpts = [ckpt_path + ckpt_file for ckpt_file in os.listdir(ckpt_path)
                   if "model_" + neural_name in ckpt_file]
    ckpt_file = None
    for ckpt in model_ckpts:
        if "opposite_" + opposite_name in ckpt:
            ckpt_file = ckpt
            if str(percentage) in ckpt:
                break
    if ckpt_file is None and len(model_ckpts) > 0:
        ckpt_file = random.choice(model_ckpts)
    return neural_proc, neural_model, ckpt_file
