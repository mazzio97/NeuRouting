import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import softmax
from torch.distributions.categorical import Categorical


class GatConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.convs = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for _ in range(conv_layers)])

    def forward(self, data):
        batch_size = data.num_graphs
        x = self.fc_node(data.x)
        edge_attr = self.fc_edge(data.edge_attr)
        x = self.bn(x)
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
        x = x.reshape((batch_size, -1, self.hidden_node_dim))
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)
        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        u_i = u_i.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(u_i, dim=1)
        return scores


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = torch.nn.GRUCell(input_dim, hidden_dim, bias=True)
        self.attn = Attention(hidden_dim)

    def evaluate(self, encoder_inputs, hx, actions):
        _input = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(2)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        log_ps = []
        entropies = []

        actions = actions.transpose(0, 1)
        for act in actions:
            hx = self.cell(_input, hx)
            p = self.attn(hx, encoder_inputs, mask)
            dist = Categorical(p)
            entropy = dist.entropy()

            log_p = dist.log_prob(act)
            log_ps.append(log_p)
            mask = mask.scatter(1, act.unsqueeze(-1).expand(mask.size(0), -1), 1)
            _input = torch.gather(encoder_inputs, 1,
                                  act.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                         encoder_inputs.size(2))
                                  ).squeeze(1)
            entropies.append(entropy)

        log_ps = torch.stack(log_ps, 1)
        entropies = torch.stack(entropies, 1)
        log_p = log_ps.sum(dim=1)
        entropy = entropies.mean(dim=1)

        return log_p, entropy

    def forward(self, encoder_inputs, hx, n_steps, greedy=False):
        _input = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(2)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        log_ps = []
        actions = []
        entropies = []

        for i in range(n_steps):
            hx = self.cell(_input, hx)
            p = self.attn(hx, encoder_inputs, mask)
            dist = Categorical(p)
            entropy = dist.entropy()

            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()

            actions.append(index)
            log_p = dist.log_prob(index)
            log_ps.append(log_p)
            entropies.append(entropy)

            mask = mask.scatter(1, index.unsqueeze(-1).expand(mask.size(0), -1), 1)
            _input = torch.gather(encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                           encoder_inputs.size(2))).squeeze(1)

        log_ps = torch.stack(log_ps, 1)
        actions = torch.stack(actions, 1)
        entropies = torch.stack(entropies, 1)
        log_p = log_ps.sum(dim=1)
        entropy = entropies.mean(dim=1)
        return actions, log_p, entropy


class EgateModel(nn.Module):
    def __init__(self, input_node_dim=5, hidden_node_dim=64, input_edge_dim=2, hidden_edge_dim=16, conv_layers=2):
        super(EgateModel, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                               conv_layers=conv_layers)
        self.decoder = Decoder(hidden_node_dim, hidden_node_dim)
        self.v1 = nn.Linear(hidden_node_dim, hidden_node_dim)
        self.v2 = nn.Linear(hidden_node_dim, 1)

    def forward(self, data_batch, steps, greedy=False):
        x = self.encoder(data_batch)
        x = x[:, 1:, :]
        pooled = x.mean(dim=1)
        actions, log_p, entropy = self.decoder(x, pooled, steps, greedy)
        v = self.v1(pooled)
        v = self.v2(v)
        v = v.squeeze(-1)
        return actions, log_p, v, entropy

    def evaluate(self, data_batch, actions):
        x = self.encoder(data_batch)
        x = x[:, 1:, :]
        pooled = x.mean(dim=1)
        log_p, entropy = self.decoder.evaluate(x, pooled, actions)
        v = self.v1(pooled)
        v = self.v2(v)
        v = v.squeeze(-1)
        return log_p, v, entropy
