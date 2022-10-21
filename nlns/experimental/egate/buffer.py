import numpy as np
import torch
from torch_geometric.data import Data, DataLoader


class Buffer:
    def __init__(self):
        super(Buffer, self).__init__()
        self.buf_nodes = []
        self.buf_edges = []
        self.buf_actions = []
        self.buf_rewards = []
        self.buf_values = []
        self.buf_log_probs = []

    def obs(self, nodes, edges, actions, rewards, log_probs, values):
        self.buf_nodes.append(nodes)
        self.buf_edges.append(edges)
        self.buf_actions.append(actions)
        self.buf_rewards.append(rewards)
        self.buf_values.append(values)
        self.buf_log_probs.append(log_probs)

    def compute_values(self, last_v=0, _lambda=1.0):
        rewards = np.array(self.buf_rewards)
        # rewards = (rewards - rewards.mean()) / rewards.std()
        pred_vs = np.array(self.buf_values)
        target_vs = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        v = last_v
        for i in reversed(range(rewards.shape[0])):
            v = rewards[i] + _lambda * v
            target_vs[i] = v
            adv = v - pred_vs[i]
            advs[i] = adv
        return target_vs, advs

    def generate_dataset(self, last_v=0, _lambda=0.99):
        target_vs, advs = self.compute_values(last_v, _lambda)
        advs = (advs - advs.mean()) / advs.std()
        l, w = target_vs.shape
        dataset = []
        for i in range(l):
            for j in range(w):
                nodes = self.buf_nodes[i][j]
                edges = self.buf_edges[i][j]
                action = self.buf_actions[i][j]
                v = target_vs[i][j]
                adv = advs[i][j]
                log_prob = self.buf_log_probs[i][j]
                edge_index = torch.LongTensor([[i, j] for i in range(len(nodes)) for j in range(len(nodes))]).T
                data = Data(x=torch.from_numpy(nodes).float(), edge_index=edge_index,
                            edge_attr=torch.from_numpy(edges).float(), v=torch.tensor([v]).float(),
                            action=torch.tensor(action).long(),
                            log_prob=torch.tensor([log_prob]).float(),
                            adv=torch.tensor([adv]).float())
                dataset.append(data)
        return dataset

    @staticmethod
    def to_dataset(_nodes, _edges):
        dataset = []
        n_graphs = len(_nodes)
        n_nodes = len(_nodes[0])
        edge_index = torch.LongTensor([[i, j] for i in range(n_nodes) for j in range(n_nodes)]).T
        for i in range(n_graphs):
            nodes = _nodes[i]
            edges = _edges[i]
            data = Data(x=torch.from_numpy(nodes).float(), edge_index=edge_index,
                        edge_attr=torch.from_numpy(edges).float())
            dataset.append(data)
        return list(DataLoader(dataset, batch_size=n_graphs))[0]
