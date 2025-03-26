import math
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from torch_geometric.data import Data

from nettcrstruc.dataset.mappings import CHAIN_TO_NUM, LETTER_TO_NUM
from nettcrstruc.utils import pdb_utils


def extract_features_from_pdb(
    pdb_path: Path,
    chain_order=["D", "E", "C", "A"],
) -> tuple:
    """Extracts relevant information from a PDB file for graph featurization.

    Args:
        pdb_path: Path to PDB file.
        chain_order: Order that chains of structure should appear in.

    Returns:
        sequence: Amino acid sequence.
        chain_id: Chain ID for each residue.
        backbone_coords: Backbone coordinates.
    """
    # Fetch structure data
    structure = strucio.load_structure(pdb_path, extra_fields=["b_factor"])

    # Merge MHC class II B chain to A
    structure = pdb_utils.merge_mhc_chains(structure)

    # Set order of chains to always be the same
    structure = pdb_utils.set_chain_order(structure, chain_order)

    sequence = "".join(
        struc.apply_residue_wise(
            structure,
            structure.res_name,
            pdb_utils.convert_3to1,
        )
    )
    chain_id = struc.apply_residue_wise(
        structure,
        structure.chain_id,
        lambda *args, **kwargs: args[0][0],
    )
    backbone_coords = struc.apply_residue_wise(
        structure,
        structure,
        pdb_utils.get_backbone_coords,
    )
    return (
        sequence,
        chain_id,
        backbone_coords,
        structure,
    )


def get_geometric_features(
    feature_path: Path,
    name: str,
    device: str,
    seq: str,
    chain_id: np.ndarray,
    coords: np.ndarray,
) -> Data:
    """Helper function for extracting or loading geometric features.

    Args:
        feature_path: Path to save/load features.
        name: Name of entry.
        device: If "cuda", preprocessing is performed on the GPU.
        seq: Amino acid sequence.
        chain_id: Chain ID for each residue.
        coords: Array with coordinates of N, Ca, C, and O atoms.

    Returns:
        features: torch_geometric.data.Data object with geometric features.
    """
    if feature_path.exists():
        features = torch.load(feature_path, map_location=torch.device(device))
    else:
        features = featurize_as_graph(
            name=name,
            seq=seq,
            coords=coords,
            chain_id=chain_id,
            top_k=30,
            num_positional_embeddings=16,
            num_rbf=16,
            device=device,
            letter_to_num=LETTER_TO_NUM,
            chain_to_num=CHAIN_TO_NUM,
        )
        torch.save(features, feature_path)
    return features


def featurize_as_graph(
    name: str,
    seq: str,
    coords: np.ndarray,
    chain_id: np.ndarray,
    top_k: int,
    num_positional_embeddings: int,
    num_rbf: int,
    device: str,
    letter_to_num: dict,
    chain_to_num: dict,
) -> Data:
    """Modified from https://github.com/jingraham/neurips19-graph-protein-design

    Transforms JSON/dictionary-style protein structures into featurized graphs.

    Args:
        name: Name of entry.
        seq: Amino acid sequence.
        coords: Array with coordinates of N, Ca, C, and O atoms.
        chain_id: Chain ID for each residue.
        top_k: Number of edges to draw per node.
        num_positional_embeddings: Number of positional embeddings.
        num_rbf: Number of radial basis function features.
        device: If "cuda", preprocessing is performed on the GPU.
        letter_to_num: Mapping of amino acid letters to integers.
        chain_to_num: Mapping of chain identifiers to integers.

    Returns Data object with:
        x: Alpha carbon coordinates, shape [n_nodes, 3].
        seq: Sequence converted to an integer tensor using `self.letter_to_num`, shape [n_nodes].
        name: Name of the protein structure.
        node_s: Node scalar features, shape [n_nodes, 6].
        node_v: Node vector features, shape [n_nodes, 3, 3].
        edge_s: Edge scalar features, shape [n_edges, 32].
        edge_v: Edge vector features, shape [n_edges, 1, 3].
        edge_index: Edge indices, shape [2, n_edges].
        mask: Node mask, where `False` indicates missing data excluded from message passing.
    """
    with torch.no_grad():
        coords = torch.as_tensor(coords, device=device, dtype=torch.float32)
        seq = torch.as_tensor(
            [letter_to_num[a] for a in seq], device=device, dtype=torch.long
        )
        chain_id = torch.as_tensor(
            [chain_to_num[c] for c in chain_id], device=device, dtype=torch.long
        )

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        pos_embeddings = positional_embeddings(
            edge_index, num_positional_embeddings, device
        )

        interchain_edges = chain_id[edge_index[0]] != chain_id[edge_index[1]]
        pos_embeddings[interchain_edges] = torch.zeros(
            (num_positional_embeddings,), device=device
        )

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)
        dihedrals = compute_dihedrals(coords)
        orientations = compute_orientations(X_ca)
        sidechains = compute_sidechains(coords)

        node_s = dihedrals
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        node_v = torch.cat([orientations, sidechains.unsqueeze(1)], dim=1)
        edge_v = _normalize(E_vectors).unsqueeze(1)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

    data = Data(
        x=X_ca,
        seq=seq,
        chain_id=chain_id,
        name=name,
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        mask=mask,
    )
    data.to(device)
    return data


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def compute_dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])

    # Lift angle representations to the circle
    return torch.cat([torch.cos(D), torch.sin(D)], 1)


def positional_embeddings(edge_index, num_embeddings, device):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


def compute_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def compute_sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec
