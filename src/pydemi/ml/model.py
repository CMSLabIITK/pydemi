# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
#
# The ChargE3Net network (Koker et al., npj Comput. Mater. 10, 161 (2024)),
# from https://github.com/AIforGreatGood/charge3net, MIT licence (LICENSE-charge3net).
# Changes for pydemi: inference on one structure at a time, so the forward passes
# take plain (unbatched) tensors instead of padded batches, and the species count
# no longer comes from ASE. Module and parameter names are unchanged, so the
# checkpoints of the original code load as they are.
"""The ChargE3Net network: atom representations, then messages from atoms to probe points."""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct
from torch import nn

NUM_SPECIES = 119  # len(ase.data.atomic_numbers): the dummy "X" plus elements 1-118

_ACT = {1: torch.nn.functional.silu, -1: torch.tanh}
_ACT_GATES = {1: torch.sigmoid, -1: torch.tanh}


def get_irreps(total_mul, lmax):
    """Irreps up to lmax, all with roughly the same multiplicity, total multiplicity total_mul.

    get_irreps(500, lmax=2) = 167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e
    """
    return [(round(total_mul / (lmax + 1) / (l * 2 + 1)), (l, p))
            for l in range(lmax + 1) for p in [-1, 1]]


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def _gate(irreps_node, irreps_node_hidden, irreps_edge_attr):
    """Gate activation (https://docs.e3nn.org/en/stable/api/nn/nn_gate.html) for one layer."""
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_node_hidden
                                if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)]
                               ).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_node_hidden
                              if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)])
    ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return Gate(irreps_scalars, [_ACT[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [_ACT_GATES[ir.p] for _, ir in irreps_gates], irreps_gated)


class E3DensityModel(nn.Module):
    def __init__(self, num_interactions=3, num_neighbors=20, mul=500, lmax=4, cutoff=4.0,
                 basis="gaussian", num_basis=10):
        super().__init__()
        self.cutoff = cutoff
        self.atom_model = E3AtomRepresentationModel(num_interactions, num_neighbors, mul=mul,
                                                    lmax=lmax, cutoff=cutoff, basis=basis,
                                                    num_basis=num_basis)
        self.probe_model = E3ProbeMessageModel(num_interactions, num_neighbors,
                                               self.atom_model.atom_irreps_sequence, mul=mul,
                                               lmax=lmax, cutoff=cutoff, basis=basis,
                                               num_basis=num_basis)


class E3AtomRepresentationModel(nn.Module):
    def __init__(self, num_interactions, num_neighbors, mul=500, lmax=4, cutoff=4.0,
                 basis="gaussian", num_basis=10):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(start=0.0, end=cutoff, number=num_basis, basis=basis,
                                 cutoff=False, normalize=True)
        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.atom_irreps_sequence = []  # output irreps of each layer, for the probe model
        self.num_species = NUM_SPECIES

        irreps_node = f"{self.num_species}x 0e"  # one-hot atomic numbers, even parity
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        for _ in range(num_interactions):
            gate = _gate(irreps_node, irreps_node_hidden, irreps_edge_attr)
            conv = Convolution(irreps_node, "0e", irreps_edge_attr, gate.irreps_in,
                               [num_basis, 100], num_neighbors)
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
            self.atom_irreps_sequence.append(irreps_node)

    def forward(self, numbers, edge_src, edge_dst, edge_vec):
        """Node features of every layer. ``numbers`` (n_atoms,) atomic numbers; edges from
        ``edge_src`` to ``edge_dst`` with vectors ``edge_vec`` = r_dst - r_src (n_edges, 3)."""
        nodes = torch.nn.functional.one_hot(numbers, num_classes=self.num_species).to(edge_vec.dtype)
        node_attr = nodes.new_ones(nodes.shape[0], 1)
        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True,
                                           normalization="component")
        edge_length_embedding = self.basis(edge_vec.norm(dim=1))
        nodes_list = []
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = gate(conv(nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding))
            nodes_list.append(nodes)
        return nodes_list


class E3ProbeMessageModel(torch.nn.Module):
    def __init__(self, num_interactions, num_neighbors, atom_irreps_sequence, mul=500, lmax=4,
                 cutoff=4.0, basis="gaussian", num_basis=10):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(start=0.0, end=cutoff, number=num_basis, basis=basis,
                                 cutoff=False, normalize=True)
        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        irreps_node = "0e"  # probes start as zero scalars
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        for i in range(num_interactions):
            gate = _gate(irreps_node, irreps_node_hidden, irreps_edge_attr)
            conv = ConvolutionOneWay(
                irreps_sender_input=atom_irreps_sequence[i], irreps_sender_attr="0e",
                irreps_receiver_input=irreps_node, irreps_receiver_attr="0e",
                irreps_edge_attr=irreps_edge_attr, irreps_node_output=gate.irreps_in,
                fc_neurons=[num_basis, 100], num_neighbors=num_neighbors)
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
        self.readout = Linear(irreps_node, "0e")

    def forward(self, atom_representation, n_probes, edge_src, edge_dst, edge_vec):
        """Density at ``n_probes`` points; edges from atom ``edge_src`` to probe ``edge_dst``,
        ``edge_vec`` = r_probe - r_atom."""
        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True,
                                           normalization="component")
        edge_length_embedding = self.basis(edge_vec.norm(dim=1))
        first = atom_representation[0]
        probes = first.new_zeros((n_probes, 1))
        probe_attr = probes.new_ones(n_probes, 1)
        atom_attr = probes.new_ones(first.shape[0], 1)
        for conv, gate, atom_nodes in zip(self.convolutions, self.gates, atom_representation):
            probes = gate(conv(atom_nodes, atom_attr, probes, probe_attr, edge_src, edge_dst,
                               edge_attr, edge_length_embedding))
        return self.readout(probes)[:, 0]


class RadialBasis(nn.Module):
    r"""e3nn.math.soft_one_hot_linspace, normalized to mean 0, std 1 over [start, end]."""

    def __init__(self, start, end, number, basis="gaussian", cutoff=False, normalize=True,
                 samples=4000):
        super().__init__()
        self.start, self.end, self.number = start, end, number
        self.basis, self.cutoff, self.normalize = basis, cutoff, normalize
        if normalize:
            with torch.no_grad():
                rs = torch.linspace(start, end, samples + 1)[1:]
                bs = soft_one_hot_linspace(rs, start, end, number, basis, cutoff)
                std, mean = torch.std_mean(bs, dim=0)
            self.register_buffer("mean", mean)
            self.register_buffer("inv_std", torch.reciprocal(std))

    def forward(self, x):
        x = soft_one_hot_linspace(x, self.start, self.end, self.number, self.basis, self.cutoff)
        if self.normalize:
            x = (x - self.mean) * self.inv_std
        return x


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """torch_scatter.scatter(src, index, dim=0, reduce="sum")."""
    out = src.new_zeros(dim_size, src.shape[1])
    return out.scatter_add_(0, index.reshape(-1, 1).expand_as(src), src)


def _mid_irreps(irreps_input, irreps_edge_attr, irreps_node_output):
    irreps_mid, instructions = [], []
    for i, (mul, ir_in) in enumerate(irreps_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_mid, p, _ = o3.Irreps(irreps_mid).sort()
    return irreps_mid, [(i1, i2, p[i_out], mode, train) for i1, i2, i_out, mode, train in instructions]


# Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
# University of California, through Lawrence Berkeley National Laboratory
# (subject to receipt of any required approvals from the U.S. Dept. of Energy),
# Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin
# and Kostiantyn Lapchevskyi. All rights reserved.
# Modified from https://github.com/e3nn/e3nn/blob/05b386177ed039156526f9c67d0d87b6c21ff5d3/e3nn/nn/models/v2103/points_convolution.py
#  - Remove torch_scatter dependency
#  - Add support for differently indexed sending/receiver nodes.
#  - Sender and receiver nodes can have different irreps.
class Convolution(torch.nn.Module):
    """Equivariant convolution between atoms."""

    def __init__(self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output,
                 fc_neurons, num_neighbors) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors
        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                              self.irreps_node_output)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                                self.irreps_node_input)
        irreps_mid, instructions = _mid_irreps(self.irreps_node_input, self.irreps_edge_attr,
                                               self.irreps_node_output)
        self.tp = TensorProduct(self.irreps_node_input, self.irreps_edge_attr, irreps_mid,
                                instructions, internal_weights=False, shared_weights=False)
        self.fc = FullyConnectedNet(fc_neurons + [self.tp.weight_numel], torch.nn.functional.silu)
        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr,
                                                self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
        weight = self.fc(edge_scalars)
        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim_size=node_input.shape[0]
                                ).div(self.num_neighbors ** 0.5)
        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)  # start small, favour self-connection
        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


class ConvolutionOneWay(torch.nn.Module):
    """Equivariant convolution from sender nodes (atoms) to differently indexed receivers (probes)."""

    def __init__(self, irreps_sender_input, irreps_sender_attr, irreps_receiver_input,
                 irreps_receiver_attr, irreps_edge_attr, irreps_node_output, fc_neurons,
                 num_neighbors) -> None:
        super().__init__()
        self.irreps_sender_input = o3.Irreps(irreps_sender_input)
        self.irreps_sender_attr = o3.Irreps(irreps_sender_attr)
        self.irreps_receiver_input = o3.Irreps(irreps_receiver_input)
        self.irreps_receiver_attr = o3.Irreps(irreps_receiver_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors
        self.sc = FullyConnectedTensorProduct(self.irreps_receiver_input,
                                              self.irreps_receiver_attr, self.irreps_node_output)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_sender_input, self.irreps_sender_attr,
                                                self.irreps_sender_input)
        irreps_mid, instructions = _mid_irreps(self.irreps_sender_input, self.irreps_edge_attr,
                                               self.irreps_node_output)
        self.tp = TensorProduct(self.irreps_sender_input, self.irreps_edge_attr, irreps_mid,
                                instructions, internal_weights=False, shared_weights=False)
        self.fc = FullyConnectedNet(fc_neurons + [self.tp.weight_numel], torch.nn.functional.silu)
        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_receiver_attr,
                                                self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_receiver_attr, "0e")

    def forward(self, sender_input, sender_attr, receiver_input, receiver_attr, edge_src,
                edge_dst, edge_attr, edge_scalars):
        weight = self.fc(edge_scalars)
        receiver_self_connection = self.sc(receiver_input, receiver_attr)
        sender_features = self.lin1(sender_input, sender_attr)
        edge_features = self.tp(sender_features[edge_src], edge_attr, weight)
        receiver_features = scatter(edge_features, edge_dst, dim_size=receiver_input.shape[0]
                                    ).div(self.num_neighbors ** 0.5)
        receiver_conv_out = self.lin2(receiver_features, receiver_attr)
        receiver_angle = 0.1 * self.lin3(receiver_features, receiver_attr)
        cos, sin = receiver_angle.cos(), receiver_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * receiver_self_connection + sin * receiver_conv_out
