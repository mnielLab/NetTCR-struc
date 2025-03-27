import concurrent
from typing import Union

import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
from structure_pipeline.utils.sequence_utils import get_cdr_from_sequence
from tqdm import tqdm


def set_chain_order(structure: struc.AtomArray, chain_order: list) -> struc.AtomArray:
    """Sets the order of chains in a structure.

    Args:
        structure: AtomArray containing the structure
        chain_order: List of chain IDs in the desired order

    Returns:
        AtomArray with chains in the desired order.
    """
    new_structure = struc.AtomArray(len(structure))
    if "b_factor" in structure.get_annotation_categories():
        new_structure.set_annotation("b_factor", np.zeros(len(new_structure)))
    new_structure.set_annotation(
        "chain_id",
        np.concatenate(
            [
                structure[structure.chain_id == chain_id].chain_id
                for chain_id in chain_order
            ]
        ),
    )
    for chain_id in chain_order:
        new_structure[new_structure.chain_id == chain_id] = structure[
            structure.chain_id == chain_id
        ]
        if "b_factor" in structure.get_annotation_categories():
            new_structure.b_factor[new_structure.chain_id == chain_id] = structure[
                structure.chain_id == chain_id
            ].b_factor

    return new_structure


def rename_chains(structure: struc.AtomArray, naming_map: dict) -> struc.AtomArray:
    """Renames chains in structure to match naming map.

    Args:
        structure: AtomArray structure object.
        naming_map: dictionary describing the renaming scheme.

    Returns:
        AtomArray object with renamed chains.
    """
    chain_ids = structure.chain_id
    structure.chain_id = np.array([naming_map[c] for c in chain_ids])
    return structure


def get_backbone_coords(residue: struc.AtomArray, axis=None) -> np.ndarray:
    """Extracts coordinates of N, Ca, C, and O atoms in residue."""
    return np.array(
        [
            residue[residue.atom_name == "N"].coord[0],
            residue[residue.atom_name == "CA"].coord[0],
            residue[residue.atom_name == "C"].coord[0],
            residue[residue.atom_name == "O"].coord[0],
        ]
    )


def get_plddt(residue: struc.AtomArray, axis=None) -> np.ndarray:
    """Extracts the B-factor (AF pLDDT) from the residue."""
    return residue[0].b_factor


def convert_3to1(residue: Union[str, np.ndarray], axis=None) -> str:
    """Convert a residue name from 3-letter to 1-letter code.

    Args:
        residue (str or ndarray): The residue name in 3-letter code.

    Returns:
        str: The residue name in 1-letter code.
    """
    if not isinstance(residue, str):
        residue = residue[0]
    return seq.ProteinSequence.convert_letter_3to1(residue)


def get_sequence_from_chain(structure: struc.AtomArray, chain_id: str) -> str:
    """Get the sequence of a chain in a structure.

    Args:
        structure (AtomArray): The structure.
        chain_id (str): The chain ID.

    Returns:
        str: The sequence of the chain in 1-letter code.
    """
    return "".join(
        struc.apply_residue_wise(
            structure[structure.chain_id == chain_id],
            structure[structure.chain_id == chain_id].res_name,
            convert_3to1,
            axis=None,
        )
    )


def make_res_id_continuous(
    structure: struc.AtomArray,
    chain_id: Union[str, list, None] = None,
) -> struc.AtomArray:
    """Make the residue IDs in a chain continuous.

    Args:
        structure (AtomArray): The structure.
        chain_id (str): The chain ID.

    Returns:
        AtomArray: The structure with continuous residue IDs.
    """
    if isinstance(chain_id, str):
        chain_id = [chain_id]
    if chain_id is None:
        chain_id = np.unique(structure.chain_id)

    # Set mask for selection
    selection_mask = np.isin(structure.chain_id, chain_id)
    selection = structure[selection_mask]

    # Get new residue IDs
    new_res_id = struc.spread_residue_wise(
        selection,
        np.arange(1, struc.get_residue_count(selection) + 1),
    )
    selection.res_id = new_res_id

    structure[selection_mask] = selection
    return structure


def merge_mhc_chains(structure: struc.AtomArray) -> struc.AtomArray:
    """Convert MHC class II B chain to A in the structure.

    Args:
        structure (AtomArray): The structure to convert.

    Returns:
        AtomArray: The structure with the MHC class II B chain converted to A.
    """
    if "B" in structure.chain_id:
        chain_id = structure.chain_id
        chain_id[chain_id == "B"] = "A"
        structure.chain_id = chain_id
    return structure


def get_clashes(
    structure: struc.AtomArray,
    chain1_id: str,
    chain2_id: str,
    clash_cutoff: float = 3.0,
    backbone_only: bool = True,
) -> np.ndarray:
    """Get the number of backbone atom clashes between two chains in a structure.

    Args:
        structure (AtomArray): The structure.
        chain1_id (str): The ID of the first chain.
        chain2_id (str): The ID of the second chain.
        clash_cutoff (float): The maximum distance for a clash.
        backbone_only (bool): Whether to only consider clashes between backbone
            atoms.

    Returns:
        np.ndarray: The number of clashes between the two chains.
    """

    structure = make_res_id_continuous(structure, chain1_id)
    structure = make_res_id_continuous(structure, chain2_id)

    # Ensure no hydrogen bonds are present
    structure = structure[structure.element != "H"]

    if backbone_only:
        structure = structure[
            (structure.atom_name == "CA")
            | (structure.atom_name == "N")
            | (structure.atom_name == "C")
            | (structure.atom_name == "O")
        ]

    dist_matrix = struc.distance(
        structure[structure.chain_id == chain1_id].coord[:, np.newaxis, :],
        structure[structure.chain_id == chain2_id].coord,
    )

    return np.sum(dist_matrix < clash_cutoff)


def get_tcr_peptide_clashes_for_structure(
    path: str,
    peptide_chain: str = "C",
    tra_chain: str = "D",
    trb_chain: str = "E",
) -> tuple:
    """Loads a structure and computes peptide-TCR clashes.

    Args:
        path (str): Path to the structure.
        peptide_chain (str): The chain ID of the peptide.
        tra_chain (str): The chain ID of the TRA.
        trb_chain (str): The chain ID of the TRB.

    Returns:
        tuple: The number of clashes between the peptide and TRA and TRB.
    """
    structure = strucio.load_structure(path)
    TRA_clashes = get_clashes(structure, peptide_chain, tra_chain)
    TRB_clashes = get_clashes(structure, peptide_chain, trb_chain)
    return TRA_clashes, TRB_clashes


def filter_clashes(
    df: pd.DataFrame,
    max_tra_clashes: int = 5,
    max_trb_clashes: int = 5,
    max_workers: int = 8,
) -> pd.DataFrame:
    paths = df["path"].tolist()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(get_tcr_peptide_clashes_for_structure, paths),
                total=len(paths),
            )
        )

    TRA_clashes, TRB_clashes = zip(*results)
    df["TRA_clashes"] = TRA_clashes
    df["TRB_clashes"] = TRB_clashes

    # Filter away structures with more than 5 clashes in either chain
    df = df[
        (df["TRA_clashes"] <= max_tra_clashes) & (df["TRB_clashes"] <= max_trb_clashes)
    ]
    return df
