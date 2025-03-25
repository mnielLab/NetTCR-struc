import math
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch_cluster
import torch_geometric

from nettcrstruc.utils import pdb_utils


def extract_features_from_pdb(
    pdb_path: Path,
    interface_top_k: int = 10,
    peptide_chain="C",
    mhc_chain="A",
    chain_order=["D", "E", "C", "A"],
) -> tuple:
    """Extracts relevant information from a PDB file for graph featurization.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        tuple: Tuple containing:
            - sequence: Amino acid sequence.
            - chain_id: Chain ID for each residue.
            - backbone_coords: Backbone coordinates.
            - interface_structure: Interface residues.
            - interface_residue_mask: Mask for interface residues.
            - vdw_radii: Van der Waals radii for each atom.
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
    interface_structure, interface_residue_mask = pdb_utils.get_interface(
        structure=structure,
        top_k=interface_top_k,
        peptide_chain_id=peptide_chain,
        mhc_chain_id=mhc_chain,
    )
    interface_sequence = "".join(
        pdb_utils.convert_3to1(res) for res in interface_structure.res_name
    )
    vdw_radii = [
        struc.info.vdw_radius_protor(atom.res_name, atom.atom_name)
        for atom in interface_structure
    ]
    vdw_radii = [
        struc.info.vdw_radius_protor(atom.res_name, atom.atom_name)
        for atom in interface_structure
    ]
    return (
        sequence,
        chain_id,
        backbone_coords,
        interface_structure,
        interface_sequence,
        interface_residue_mask,
        vdw_radii,
    )


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


def concatenate_vector_features(features: list) -> torch.Tensor:
    """Pads and concatenates a list of 2 dim features.

    Args:
        features: List of 2 dim features.

    Returns:
        torch.Tensor: Padded and concatenated features.
    """
    padding_length = max(feature.shape[-1] for feature in features)
    padded_tensors = []
    for feature in features:
        padded_tensors.append(
            torch.nn.functional.pad(
                feature,
                (0, padding_length - feature.shape[-1]),
                mode="constant",
                value=0.0,
            )
        )
        if padded_tensors[-1].dim() != 3:
            padded_tensors[-1] = padded_tensors[-1].unsqueeze(-2)

    return torch.cat(padded_tensors, dim=-2)


class SimpleGraphDatasetPreprocessor(data.Dataset):
    def __init__(
        self,
        edge_distance_cutoff: float,
    ):
        super(SimpleGraphDatasetPreprocessor, self).__init__()

        self.edge_distance_cutoff = edge_distance_cutoff
        self.atchley_map = {
            "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
            "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
            "D": [1.050, 0.302, -3.656, -0.259, -3.242],
            "E": [1.357, -1.453, 1.477, 0.113, -0.837],
            "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
            "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
            "H": [0.336, -0.417, -1.673, -1.474, -0.078],
            "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
            "K": [1.831, -0.561, 0.533, -0.277, 1.648],
            "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
            "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
            "N": [0.945, 0.828, 1.299, -0.169, 0.933],
            "P": [0.189, 2.081, -1.628, 0.421, -1.392],
            "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
            "R": [1.538, -0.055, 1.502, 0.440, 2.897],
            "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
            "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
            "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
            "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
            "Y": [0.260, 0.830, 3.097, -0.838, 1.512],
        }
        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
        }
        self.chain_to_num = {
            "A": 0,
            "C": 1,
            "D": 2,
            "E": 3,
        }

    def featurize_as_graph(
        self,
        coords: np.array,
        seq: str,
        chain_id: np.array,
        name: str,
    ) -> torch_geometric.data.Data:
        # Compute the distance matrix
        dist_mat = struc.distance(coords[:, np.newaxis], coords)

        edge_index = torch.tensor(
            np.array(np.nonzero(dist_mat < self.edge_distance_cutoff)), dtype=torch.long
        )

        # Remove self loops
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        edge_attr = torch.tensor(
            dist_mat[dist_mat < self.edge_distance_cutoff], dtype=torch.float32
        )

        x = torch.tensor([self.atchley_map[aa] for aa in seq], dtype=torch.float32)

        seq = torch.tensor([self.letter_to_num[aa] for aa in seq], dtype=torch.long)
        chain_id = torch.tensor(
            [self.chain_to_num[c] for c in chain_id], dtype=torch.long
        )
        # Node feature
        return torch_geometric.data.Data(
            x=x,
            seq=seq,
            chain_id=chain_id,
            edge_index=edge_index,
            edge_attr=edge_attr,
            name=name,
        )


class ProteinGraphDatasetPreprocessor(data.Dataset):
    """
    Modified from https://github.com/jingraham/neurips19-graph-protein-design

    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the
    manuscript.

    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing

    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    """

    def __init__(
        self,
        top_k=30,
        num_rbf=16,
        num_positional_embeddings=16,
        device="cpu",
        file_list=[],
    ):
        super(ProteinGraphDatasetPreprocessor, self).__init__()

        self.num_positional_embeddings = num_positional_embeddings
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.device = device
        self.file_list = file_list

        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
        }
        self.chain_to_num = {
            "A": 0,
            "C": 1,
            "D": 2,
            "E": 3,
        }
        self.element_to_num = {
            "C": 0,
            "N": 1,
            "O": 2,
            "S": 3,
        }

        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

    def __len__(self):
        return len(self.file_list)

    def _featurize_as_graph(
        self,
        name,
        seq,
        coords,
        extra_node_scalar_features,
        extra_node_vector_features,
        extra_edge_scalar_features,
        extra_edge_vector_features,
        chain_id,
    ):
        with torch.no_grad():
            coords = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
            seq = torch.as_tensor(
                [self.letter_to_num[a] for a in seq],
                device=self.device,
                dtype=torch.long,
            )
            chain_id = torch.as_tensor(
                [self.chain_to_num[c] for c in chain_id],
                device=self.device,
                dtype=torch.long,
            )
            extra_node_scalar_features = [
                torch.as_tensor(f, device=self.device, dtype=torch.float32)
                for f in extra_node_scalar_features
            ]
            extra_node_vector_features = [
                torch.as_tensor(f, device=self.device, dtype=torch.float32)
                for f in extra_node_vector_features
            ]
            extra_edge_scalar_features = [
                torch.as_tensor(f, device=self.device, dtype=torch.float32)
                for f in extra_edge_scalar_features
            ]
            extra_edge_vector_features = [
                torch.as_tensor(f, device=self.device, dtype=torch.float32)
                for f in extra_edge_vector_features
            ]

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)

            # Set distance of inter-chain distances to zero
            interchain_edges = chain_id[edge_index[0]] != chain_id[edge_index[1]]
            pos_embeddings[interchain_edges] = torch.zeros(
                (self.num_positional_embeddings,), device=self.device
            )

            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            # Concatenate scalar features
            node_s = torch.cat(
                [dihedrals]
                + [
                    feature.unsqueeze(-1) if feature.dim() == 1 else feature
                    for feature in extra_node_scalar_features
                ],
                dim=-1,
            )
            edge_s = torch.cat(
                [rbf, pos_embeddings]
                + [
                    (
                        feature[edge_index[0], edge_index[1]].unsqueeze(-1)
                        if feature.dim() == 2
                        else feature[edge_index[0], edge_index[1]]
                    )
                    for feature in extra_edge_scalar_features
                ],
                dim=-1,
            )

            # Concatenate vector features
            node_v = concatenate_vector_features(
                [orientations, sidechains] + extra_node_vector_features
            )
            edge_v = concatenate_vector_features(
                [_normalize(E_vectors)]
                + [
                    feature[edge_index[0], edge_index[0]]
                    for feature in extra_edge_vector_features
                ]
            )
            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        return torch_geometric.data.Data(
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

    def _dihedrals(self, X, eps=1e-7):
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
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(
        self, edge_index, num_embeddings=None, period_range=[2, 1000]
    ):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    def get(self, idx) -> tuple:
        pdb_path = self.file_list[idx]
        (
            sequence,
            chain_id,
            backbone_coords,
        ) = extract_features_from_pdb(pdb_path)

        complex = self._featurize_as_graph(
            pdb_path.stem,
            sequence,
            backbone_coords,
            [],
            [],
            [],
            [],
            chain_id,
        )
        return complex

    def __getitem__(self, idx):
        data = self.get(idx)
        return [data[0], 1, 1]  # Add dummy variables
