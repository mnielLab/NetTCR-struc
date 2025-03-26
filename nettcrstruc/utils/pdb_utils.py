import concurrent
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
from Bio.PDB import Entity, MMCIFParser, PDBParser, Polypeptide
from structure_pipeline.utils.sequence_utils import get_cdr_from_sequence
from tqdm import tqdm


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


def concatenate(atoms):
    """
    Adapter from the biotite package. Concatenate multiple :class:`AtomArray` or :class:`AtomArrayStack` objects into
    a single :class:`AtomArray` or :class:`AtomArrayStack`, respectively.

    Parameters
    ----------
    atoms : iterable object of AtomArray or AtomArrayStack
        The atoms to be concatenated.
        :class:`AtomArray` cannot be mixed with :class:`AtomArrayStack`.

    Returns
    -------
    concatenated_atoms : AtomArray or AtomArrayStack
        The concatenated atoms, i.e. its ``array_length()`` is the sum of the
        ``array_length()`` of the input ``atoms``.

    Notes
    -----
    The following rules apply:

    - Only the annotation categories that exist in all elements are transferred.
    - The box of the first element that has a box is transferred, if any.
    - The bonds of all elements are concatenated, if any element has associated bonds.
      For elements without a :class:`BondList` an empty :class:`BondList` is assumed.

    Examples
    --------

    >>> atoms1 = array([
    ...     Atom([1,2,3], res_id=1, atom_name="N"),
    ...     Atom([4,5,6], res_id=1, atom_name="CA"),
    ...     Atom([7,8,9], res_id=1, atom_name="C")
    ... ])
    >>> atoms2 = array([
    ...     Atom([1,2,3], res_id=2, atom_name="N"),
    ...     Atom([4,5,6], res_id=2, atom_name="CA"),
    ...     Atom([7,8,9], res_id=2, atom_name="C")
    ... ])
    >>> print(concatenate([atoms1, atoms2]))
                1      N                1.000    2.000    3.000
                1      CA               4.000    5.000    6.000
                1      C                7.000    8.000    9.000
                2      N                1.000    2.000    3.000
                2      CA               4.000    5.000    6.000
                2      C                7.000    8.000    9.000
    """
    # Ensure that the atoms can be iterated over multiple times
    if not isinstance(atoms, Sequence):
        atoms = list(atoms)

    length = 0
    depth = None
    element_type = None
    common_categories = set(atoms[0].get_annotation_categories())
    box = None
    for element in atoms:
        if element_type is None:
            element_type = type(element)
        else:
            if not isinstance(element, element_type):
                raise TypeError(
                    f"Cannot concatenate '{type(element).__name__}' "
                    f"with '{element_type.__name__}'"
                )
        length += element.array_length()
        if isinstance(element, struc.AtomArrayStack):
            if depth is None:
                depth = element.stack_depth()
            else:
                if element.stack_depth() != depth:
                    raise IndexError("The stack depths are not equal")
        common_categories &= set(element.get_annotation_categories())
        if element.box is not None and box is None:
            box = element.box

    if element_type == struc.AtomArray:
        concat_atoms = struc.AtomArray(length)
    elif element_type == struc.AtomArrayStack:
        concat_atoms = struc.AtomArrayStack(depth, length)
    concat_atoms.coord = np.concatenate([element.coord for element in atoms], axis=-2)
    for category in common_categories:
        concat_atoms.set_annotation(
            category,
            np.concatenate(
                [element.get_annotation(category) for element in atoms], axis=0
            ),
        )
    concat_atoms.box = box

    return concat_atoms


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


def get_sequences_from_pdb(pdb_file: Union[Path, Entity.Entity]) -> dict:
    """Returns a dictionary of sequences from a PDB file.

    Args:
        pdb_file: Path to PDB file or loaded PDB structure.

    Returns:
        Dictionary of sequences.
    """
    if isinstance(pdb_file, Path) or isinstance(pdb_file, str):
        if pdb_file.suffix == ".cif":
            parser = MMCIFParser()
            structure = parser.get_structure("cif", pdb_file)
        else:
            parser = PDBParser()
            structure = parser.get_structure("pdb", pdb_file)
    sequences = {}
    for chain in structure[0].get_chains():
        seq = []
        for residue in chain.get_residues():
            if Polypeptide.is_aa(residue, standard=True):
                seq.append(Polypeptide.protein_letters_3to1[residue.get_resname()])
            else:
                seq.append("-")
        sequences[chain.id] = "".join(seq)
    return sequences


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


def select_within_com(
    structure: struc.AtomArray,
    distance_threshold: float,
) -> struc.AtomArray:
    """Selects all atoms within a given distance from the center of mass.

    Args:
        structure (AtomArray): The structure to be selected from.
        distance_threshold (float): The distance threshold.

    Returns:
        AtomArray: The selected atoms.
    """
    center_of_mass = struc.mass_center(structure)
    distances = np.sqrt(np.sum((structure.coord - center_of_mass) ** 2, axis=1))
    atom_mask = distances <= distance_threshold
    return structure[atom_mask]


def get_interface(
    structure: struc.AtomArray,
    top_k: int,
    peptide_chain_id: str = "C",
    mhc_chain_id: str = "A",
) -> tuple:
    """Selects atoms for top K closest residues to each peptide residue.

    Args:
        structure: AtomArray containing the structure
        top_k: Number of residues to select
        peptide_chain_id: Chain ID of the peptide

    Returns:
        Tuple with AtomArray containing the interface and a boolean mask for interface.
    """
    # Extract peptide CA
    peptide_ca = structure[
        (structure.chain_id == peptide_chain_id) & (structure.atom_name == "CA")
    ]

    # Get K interfacing receptor residues for each chain
    interface_ca = []
    for chain_id in np.unique(structure.chain_id):
        if chain_id == mhc_chain_id:
            continue
        if chain_id == peptide_chain_id:
            interface_ca.append(peptide_ca)
        chain_ca = structure[
            (structure.chain_id == chain_id) & (structure.atom_name == "CA")
        ]
        distance_matrix = struc.distance(
            chain_ca.coord[:, np.newaxis, :],
            peptide_ca.coord,
        )
        for i in range(len(peptide_ca)):
            top_k_ca = np.argsort(distance_matrix[:, i])[:top_k]
            interface_ca.append(chain_ca[top_k_ca])

    interface_ca = np.concatenate(interface_ca)
    interface = structure[np.isin(structure.res_id, [ca.res_id for ca in interface_ca])]

    interface_residue_mask = np.zeros(struc.get_residue_count(structure), dtype=bool)
    interface_residue_mask[np.unique(interface.res_id) - 1] = True

    return (interface, interface_residue_mask)


def get_principal_axes(coord: np.array, scale_factor: int = 9) -> np.array:
    """Computes the principal axes of a molecule.

    Args:
        coord (np.array): The coordinates of the molecule.
        scale_factor (int): The factor by which to scale the principal axes.

    Returns:
        np.array: The principal axes of the molecule.
    """
    # Compute geometric center and center the coordinates
    center = np.mean(coord, 0)
    coord = coord - center

    # Compute principal axis matrix
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)

    # Order eigen vectors
    order = np.argsort(e_values)
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    # Center axes to the geometric center of the molecule and scale by oder
    point1 = 3 * scale_factor * axis1 + center
    point2 = 2 * scale_factor * axis2 + center
    point3 = 1 * scale_factor * axis3 + center

    return point1, point2, point3


def swap_chain_names(
    structure: struc.AtomArray,
    chain1_id: str,
    chain2_id: str,
) -> struc.AtomArray:
    """Swaps the chain names in a structure.

    Args:
        structure: AtomArray containing the structure
        chain1_id: Chain ID to swap
        chain2_id: Chain ID to swap

    Returns:
        AtomArray with swapped chain names.
    """
    chain_id_array = structure.chain_id

    # Set tmp chain names
    chain_id_array[chain_id_array == chain1_id] = "_X"
    chain_id_array[chain_id_array == chain2_id] = "_Y"

    # Swap chain names
    chain_id_array[chain_id_array == "_X"] = chain2_id
    chain_id_array[chain_id_array == "_Y"] = chain1_id

    structure.set_annotation("chain_id", chain_id_array)
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


def get_residue_mask(subset_structure, structure):
    """Make a masking array for the residues that are in both structures, that
    have a CA atom.

    Args:
        structure_1 (AtomArray): The first structure
        structure_2 (AtomArray): The second structure
    """
    mask = []
    chain_ids, idx = np.unique(structure.chain_id, return_index=True)
    for chain_id in chain_ids[np.argsort(idx)]:
        full_sequence = get_sequence_from_chain(
            structure[structure.chain_id == chain_id],
            chain_id,
        )
        subset_sequence = get_sequence_from_chain(
            subset_structure[subset_structure.chain_id == chain_id],
            chain_id,
        )

        # Find the index where the subset sequence is in the full sequence
        idx = full_sequence.find(subset_sequence)
        if idx == -1:
            chain_mask = np.ones(len(full_sequence), dtype=bool)
        else:
            # Make a mask for the residues that are in both structures
            chain_mask = np.zeros(len(full_sequence), dtype=bool)
            chain_mask[idx : idx + len(subset_sequence)] = True

        mask.extend(chain_mask)

    # Convert from residue mask to atom mask
    return struc.spread_residue_wise(structure, np.array(mask))


def get_atom_mask(subset_structure, structure):
    # for each residue, filter structure to only contain atoms in the corresponding residue in subset_structure
    atom_mask = []
    for chain_id in np.array(list(dict.fromkeys(structure.chain_id))):
        subset_chain = subset_structure[subset_structure.chain_id == chain_id]
        chain = structure[structure.chain_id == chain_id]

        for res_id in subset_chain[subset_chain.atom_name == "CA"].res_id:
            subset_residue = subset_chain[subset_chain.res_id == res_id]
            residue_mask = np.zeros(len(chain[chain.res_id == res_id]), dtype=bool)
            for i, atom in enumerate(chain[chain.res_id == res_id]):
                if atom.atom_name in set(subset_residue.atom_name):
                    residue_mask[i] = True
            atom_mask.extend(residue_mask)

    return atom_mask


def calculate_rmsd(
    structure_1: struc.AtomArray,
    structure_2: struc.AtomArray,
    super_imposition_chain: str = "A",
) -> float:
    """Calculate the RMSD between two structures after superimpoisition.

    Args:
        structure_1 (AtomArray): The first structure
        structure_2 (AtomArray): The second structure
    """
    return struc.rmsd(
        struc.superimpose(
            structure_2,
            structure_1,
            atom_mask=structure_1.chain_id == super_imposition_chain,
        )[0],
        structure_2,
    )


def filter_tcr_pmhc_mutual_distance(
    structure: struc.AtomArray,
    distance_filter: float,
) -> struc.AtomArray:

    is_pmhc = np.logical_or(structure.chain_id == "A", structure.chain_id == "C")
    is_tcr = np.logical_or(structure.chain_id == "D", structure.chain_id == "E")
    pmhc_indices = np.where(is_pmhc)[0]
    tcr_indices = np.where(is_tcr)[0]

    # Get pairwise distances between residues
    distances = struc.distance(structure.coord[:, np.newaxis], structure.coord)
    pmch_within_14A = np.any(
        distances[pmhc_indices][:, tcr_indices] <= distance_filter,
        axis=1,
    )
    tcr_within_14A = np.any(
        distances[tcr_indices][:, pmhc_indices] <= distance_filter,
        axis=1,
    )

    pmch_filtered_indices = pmhc_indices[pmch_within_14A]
    tcr_filtered_indices = tcr_indices[tcr_within_14A]

    # Combine filtered indices
    combined_indices = np.concatenate([pmch_filtered_indices, tcr_filtered_indices])
    return structure[combined_indices], combined_indices


def filter_tcr_pmhc_center_radius(
    structure: struc.AtomArray,
    distance_filter: float,
    peptide_chain: str = "C",
) -> struc.AtomArray:
    """Filters a structures based on distance from geometric center.
    The peptide will be included in all cases. Atoms for all residues with
    <=CA-center distance are included.

    Args:
        structure: The structure to filter.
        distance_filter: The distance cutoff from the geometric center.
    Returns:
        The filtered structure.
    """
    ca_structure = structure[structure.atom_name == "CA"]
    original_indices = ca_structure.res_id
    original_peptide_indices = ca_structure[
        ca_structure.chain_id == peptide_chain
    ].res_id

    # Make res id contiunous
    structure = make_res_id_continuous(structure)

    center = np.mean(ca_structure.coord, axis=0)
    distances = np.linalg.norm(ca_structure.coord - center, axis=1)

    # Get indices of atoms within the distance filter
    residues_within_distance = distances <= distance_filter

    # Ensure peptide residues are always True
    residues_within_distance[np.isin(ca_structure.res_id, original_peptide_indices)] = (
        True
    )
    residue_indices = np.where(residues_within_distance)[0]

    # Filter to interface residues
    structure = structure[np.isin(structure.res_id, residue_indices + 1)]

    # Replace the res_id with the original indices
    structure.res_id = struc.spread_residue_wise(
        structure, original_indices[residue_indices]
    )  # Note, do not +1 here

    return structure, residue_indices


def has_backbone_atoms(structure: struc.AtomArray) -> bool:
    """Check if a structure has all backbone atoms."""
    try:
        _ = struc.apply_residue_wise(
            structure,
            structure,
            get_backbone_coords,
        )
        return True
    except IndexError:
        return False


def remove_incomplete_truncating_residues(
    structure: struc.AtomArray,
) -> struc.AtomArray:
    """Remove incomplete truncating residues."""
    original_length = len(structure)
    for chain_id in np.unique(structure.chain_id):
        chain = structure[structure.chain_id == chain_id]

        # Check first and last residue of chain
        first_residue = chain[chain.res_id == chain[0].res_id]
        last_residue = chain[chain.res_id == chain[-1].res_id]

        # Check if first residue is missing atoms
        if not has_backbone_atoms(first_residue):
            structure = structure[
                ~np.logical_and(
                    structure.res_id == first_residue.res_id[0],
                    structure.chain_id == chain_id,
                )
            ]
            print(
                f"Removed first residue {first_residue.res_id[0]} from chain {chain_id}"
            )

        # Check if last residue is missing atoms
        if not has_backbone_atoms(last_residue):
            structure = structure[
                ~np.logical_and(
                    structure.res_id == last_residue.res_id[0],
                    structure.chain_id == chain_id,
                )
            ]
            print(
                f"Removed last residue {last_residue.res_id[0]} from chain {chain_id}"
            )

    # assert that no more than 2 * number of chains residues were removed
    assert original_length - len(structure) <= 2 * len(np.unique(structure.chain_id))

    return structure


def get_atom_intersection(structure_1, structure_2):
    for chain_id in np.unique(structure_2.chain_id):
        structure_2 = make_res_id_continuous(structure_2, chain_id)
        structure_1 = make_res_id_continuous(structure_1, chain_id)

    # Get the residue indices that are in both structures
    residue_mask = get_residue_mask(structure_2, structure_1)
    structure_1 = structure_1[residue_mask]

    residue_mask = get_residue_mask(structure_1, structure_2)
    structure_2 = structure_2[residue_mask]

    # Make res id identical to that of structure_2
    for c in np.array(list(dict.fromkeys(structure_1.chain_id))):
        structure_1 = make_res_id_continuous(structure_1, c)
        structure_2 = make_res_id_continuous(structure_2, c)

    atom_mask = get_atom_mask(structure_2, structure_1)
    structure_1 = structure_1[atom_mask]

    atom_mask = get_atom_mask(structure_1, structure_2)
    structure_2 = structure_2[atom_mask]

    return structure_1, structure_2


def superimpose_on_pmhc(fixed, mobile):
    fixed_pmhc = fixed[
        np.logical_or(
            fixed.chain_id == "A",
            fixed.chain_id == "C",
        )
    ]
    mobile_pmhc = mobile[
        np.logical_or(
            mobile.chain_id == "A",
            mobile.chain_id == "C",
        )
    ]

    _, transformation = struc.superimpose(
        fixed_pmhc[fixed_pmhc.atom_name == "CA"],
        mobile_pmhc[mobile_pmhc.atom_name == "CA"],
    )

    return transformation.apply(mobile)


def superimpose_on_mhc(fixed, mobile):
    _, transformation = struc.superimpose(
        fixed[(fixed.chain_id == "A") & (fixed.atom_name == "CA")],
        mobile[(mobile.chain_id == "A") & (mobile.atom_name == "CA")],
    )
    return transformation.apply(mobile)


def clean_structure(structure):
    """Remove hetero atoms and incomplete residues."""
    structure = structure[structure.hetero == False]

    # Remove residues that dont have CA atoms
    for chain_id in np.unique(structure.chain_id):
        ca_res_id = np.unique(
            structure[
                (structure.atom_name == "CA") & (structure.chain_id == chain_id)
            ].res_id
        )

        # Slice to (ca_res_id and chain_id) OR not chain_id
        structure = structure[
            np.logical_or(
                np.isin(structure.res_id, ca_res_id) & (structure.chain_id == chain_id),
                structure.chain_id != chain_id,
            )
        ]

        structure[np.isin(structure.res_id, ca_res_id)]

    return structure


def filter_to_proximal_residues(structure_1, structure_2, distance_threshold):
    """
    Filters structure_2 to residues that are within the distance threshold
    of structure_1 and includes structure_1 in the output.

    Parameters:
    structure_1 : AtomArray or AtomArrayStack
        The first structure to compare.
    structure_2 : AtomArray or AtomArrayStack
        The second structure to filter.
    distance_threshold : float
        The distance threshold to consider proximity.

    Returns:
    AtomArray
        Combined structure containing structure_1 and filtered structure_2.
    """
    # Compute pairwise distances between all atoms in structure_1 and structure_2
    distances = struc.distance(structure_1.coord[:, np.newaxis], structure_2.coord)

    # Find atoms in structure_2 that are within the distance threshold
    within_threshold = np.any(distances <= distance_threshold, axis=0)

    # Get corresponding residue IDs
    proximal_residues = set(structure_2[within_threshold].res_id)

    # Filter structure_2 for residues in the proximal set
    mask = [res_id in proximal_residues for res_id in structure_2.res_id]

    return concatenate([structure_1, structure_2[mask]])


def clean_for_rmsd_calculation(structure_1, structure_2, backbone_only=False):

    structure_1 = clean_structure(structure_1)
    structure_2 = clean_structure(structure_2)

    if backbone_only:
        structure_1 = structure_1[np.isin(structure_1.atom_name, ["N", "CA", "C", "O"])]
        structure_2 = structure_2[np.isin(structure_2.atom_name, ["N", "CA", "C", "O"])]

    # Set order of chains to be the same
    structure_1 = set_chain_order(structure_1, ["D", "E", "A", "C"])
    structure_2 = set_chain_order(structure_2, ["D", "E", "A", "C"])

    # Subset structures to intersection of atoms
    return get_atom_intersection(structure_1, structure_2)


def calculate_cdr_rmsd(
    structure_1,
    structure_2,
    cdr: str,
    chain: str,
    backbone_only: bool = False,
    interface_threshold_structure_1: float = None,
) -> float:
    cdr_number = int(cdr[1])

    structure_1, structure_2 = clean_for_rmsd_calculation(
        structure_1, structure_2, backbone_only
    )

    # Filter distance from peptide
    if interface_threshold_structure_1 is not None:
        # Use GT structure to define interface threshold
        structure_1_proxmial_res_id = filter_to_proximal_residues(
            structure_1[structure_1.chain_id == "C"],
            structure_1[structure_1.chain_id == chain],
            interface_threshold_structure_1,
        )

        # Find the corresponding residues in structure_2 TODO check if we can just use structuer_1_proxmial

    # Superimpose structures on pMHC atoms
    superimposed_structure = superimpose_on_pmhc(
        structure_1,
        structure_2,
    )

    # Filter to TCR chain
    structure_1 = structure_1[structure_1.chain_id == chain]
    superimposed_structure = superimposed_structure[
        superimposed_structure.chain_id == chain
    ]

    # Get CDR indices for structure 1
    tcr_sequence_structure_1 = get_sequence_from_chain(structure_1, chain)
    cdr_indices_structure_1 = np.array(
        get_cdr_from_sequence(tcr_sequence_structure_1, cdr_number)[1]
    )

    # Get CDR indices for structure 2
    tcr_sequence_structure_2 = get_sequence_from_chain(structure_2, chain)
    cdr_indices_structure_2 = np.array(
        get_cdr_from_sequence(tcr_sequence_structure_2, cdr_number)[1]
    )

    # Apply CDR residue indices atom-wise
    cdr_indices_structure_1 = np.where(
        np.isin(structure_1.res_id, cdr_indices_structure_1 + 1)
    )[0]

    cdr_indices_structure_2 = np.where(
        np.isin(superimposed_structure.res_id, cdr_indices_structure_2 + 1)
    )[0]

    # Get CDR res IDS
    structure_1_ids = structure_1[cdr_indices_structure_1].res_id
    superimposed_structure_ids = superimposed_structure[cdr_indices_structure_2].res_id

    # Compute CDR RMSDc
    return struc.rmsd(
        structure_1[np.isin(structure_1.res_id, structure_1_ids)],
        superimposed_structure[
            np.isin(superimposed_structure.res_id, superimposed_structure_ids)
        ],
    )


def calculate_peptide_rmsd(structure_1, structure_2, backbone_only=False):
    # Clean structure
    structure_1, structure_2 = clean_for_rmsd_calculation(
        structure_1, structure_2, backbone_only
    )

    # superimpose on MHC atoms
    superimposed_structure = superimpose_on_mhc(
        structure_1,
        structure_2,
    )

    # calculate peptide RMSD
    return struc.rmsd(
        structure_1[structure_1.chain_id == "C"],
        superimposed_structure[superimposed_structure.chain_id == "C"],
    )


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
