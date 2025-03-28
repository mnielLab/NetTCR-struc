import torch
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean


class GVPMQA(nn.Module):
    """
    Modified from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py

    GVP-GNN for DockQ regression as described in manuscript.
    """

    def __init__(
        self,
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        num_layers=3,
        drop_rate=0.1,
        out_dim=1,
        vector_gating=False,
    ):
        super(GVPMQA, self).__init__()

        self.W_s = nn.Embedding(20, 20)
        self.W_c = nn.Embedding(4, 4)

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(
                node_in_dim,
                node_h_dim,
                activations=(None, None),
                vector_gate=vector_gating,
            ),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(
                edge_in_dim,
                edge_h_dim,
                activations=(None, None),
                vector_gate=vector_gating,
            ),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_h_dim, edge_h_dim, drop_rate=drop_rate, vector_gate=vector_gating
            )
            for _ in range(num_layers)
        )

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0), vector_gate=vector_gating)
        )

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, out_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_V = (x.node_s, x.node_v)
        h_E = (x.edge_s, x.edge_v)

        seq_embedding = self.W_s(x.seq)
        chain_embedding = self.W_c(x.chain_id)
        h_V = (torch.cat([h_V[0], seq_embedding, chain_embedding], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, x.edge_index, h_E)
        out = self.W_out(h_V)

        # Pool node dimension
        out = scatter_mean(out, x.batch, dim=0)

        return self.sigmoid(self.dense(out).squeeze(-1) + 0.5)
