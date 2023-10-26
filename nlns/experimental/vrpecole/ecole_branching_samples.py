import gzip
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import List

import ecole
from ecole.core.observation import NodeBipartite
from ecole.environment import Branching

from instances import VRPInstance, VRPModelSCIP
from explore_strong_branch import ExploreThenStrongBranch


def generate_branching_samples(instances: List[VRPInstance], n_samples_instance: int, folder: str = "samples", seed: int = 0):
    Path(folder).mkdir(exist_ok=True)
    with Pool(os.cpu_count()) as pool:
        results = pool.map(_generate_branching_samples_instance,
                           [(idx, inst, n_samples_instance, folder, seed) for idx, inst in enumerate(instances)])
        pool.close()
        pool.join()
    return results


def _generate_branching_samples_instance(args):
    idx, instance, n_samples, folder, seed = args
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0}

    # Note how we can tuple observation functions to return complex state information
    branch_env = Branching(observation_function=(ExploreThenStrongBranch(expert_probability=0.05),
                                                 NodeBipartite()),
                           scip_params=scip_parameters)
    # This will seed the environment for reproducibility
    branch_env.seed(seed + idx)

    scip_model = ecole.scip.Model.from_pyscipopt(VRPModelSCIP(instance))
    observation, action_set, _, done, _ = branch_env.reset(scip_model)
    data = []
    while len(data) < n_samples:
        (scores, scores_are_expert), node_observation = observation

        action = action_set[scores[action_set].argmax()]

        # Only save samples if they are coming from the expert (strong branching)
        if scores_are_expert:
            data.append([node_observation, action, action_set, scores])
            print(f"Instance {idx}, {len(data)} samples collected so far")

        observation, action_set, _, _, _ = branch_env.step(action)

    filename = f'{folder}/instance_{idx}.pkl'
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

    return filename
