import gzip
import pickle

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class BipartiteNodeData(Data):
    """
    This class encode a node bipartite graph observation as returned by the `vrpecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = torch.FloatTensor(constraint_features)
        self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        self.edge_attr = torch.FloatTensor(edge_features).unsqueeze(1)
        self.variable_features = torch.FloatTensor(variable_features)
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, instance_files, n_samples_instance):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.instance_files = instance_files
        self.n_samples_instance = n_samples_instance

    def len(self):
        return len(self.instance_files * self.n_samples_instance)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        instance_idx = index // self.n_samples_instance
        sample_idx = index % self.n_samples_instance
        with gzip.open(self.instance_files[instance_idx], 'rb') as f:
            samples = pickle.load(f)
        sample_observation, sample_action, sample_action_set, sample_scores = samples[sample_idx]

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(sample_observation.row_features, sample_observation.edge_features.indices,
                                  sample_observation.edge_features.values, sample_observation.column_features,
                                  candidates, candidate_choice, candidate_scores)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.row_features.shape[0] + sample_observation.column_features.shape[0]

        return graph
