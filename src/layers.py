"""Classes for SimGNN modules."""
"""source:https://github.com/benedekrozemberczki/SimGNN/tree/master/src"""
import torch
import math
from torch_geometric.utils import add_remaining_self_loops

from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
class GCNConv(MessagePassing):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.constant(self.bias,0)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == 'inf'] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation

class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3, -1))
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
