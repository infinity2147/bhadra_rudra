"""TGN modules (Rossi et al. 2020), adapted for FRAUD classification.

The decoder is a single-logit fraud head over concat(Z_src, Z_dst) — not a
link-existence two-tower scorer. Memory/attention are the standard PyG TGN
primitives. Import torch lazily-safe (module import needs torch, so only import
this in a process that has already trained XGBoost, per the pipeline ordering).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    TGNMemory, IdentityMessage, LastAggregator,
)


class GraphAttentionEmbedding(nn.Module):
    """Temporal-attention embedding: TransformerConv over recent neighbors,
    with relative-time encoding concatenated onto each edge's message."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class FraudDecoder(nn.Module):
    """MLP on concat(Z_src, Z_dst) -> a single fraud logit per edge."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h)


def build_tgn(num_nodes: int, msg_dim: int, memory_dim: int = 100,
              time_dim: int = 100, embedding_dim: int = 100):
    memory = TGNMemory(
        num_nodes, msg_dim, memory_dim, time_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    )
    gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, msg_dim, memory.time_enc)
    decoder = FraudDecoder(embedding_dim)
    return memory, gnn, decoder
