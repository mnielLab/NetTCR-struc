import concurrent.futures
import json
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nettcrstruc.utils import pdb_utils
from nettcrstruc.utils.pdb_utils import get_tcr_peptide_clashes_for_structure
from nettcrstruc.utils.sequence_utils import get_cdr_from_sequence, get_cdr_indices


def get_alphafold_rankings(run_dir: Path) -> None:
    """Get metrics for an AlphaFold run directory."""
    metrics = pd.read_csv(
        run_dir / "model_scores.txt",
        sep="\t",
        names=["conf", "pLDDT", "pTM", "ipTM"],
    )
    if metrics.isnull().values.any():
        metrics = pd.read_csv(
            run_dir / "model_scores.txt",
            sep=",",
            names=["conf", "pLDDT", "pTM", "ipTM"],
        )

    # Map ranked candidates to metrics
    metrics["path"] = [
        str(run_dir / f"ranked_{i}.pdb")
        for i in range(len(list(run_dir.glob("ranked_*.pdb"))))
    ]
    return metrics


def map_ranking_to_model_order(run_dir: Path) -> None:
    """Maps ranking to model order."""
    rankings = json.load(open(run_dir / "ranking_debug.json", "r"))["order"]
    return pd.DataFrame({"rank": range(len(rankings)), "order": rankings})


def filter_clashes(df, max_tra_clashes=5, max_trb_clashes=5):
    """Filter structures based on TCR-peptide clashes.

    Args:
        df (pd.DataFrame): DataFrame with paths for TCR-pMHC structural models.
        max_tra_clashes (int): Maximum number of TRA clashes allowed.
        max_trb_clashes (int): Maximum number of TRB clashes allowed.
    """
    paths = df["path"].tolist()
    with concurrent.futures.ProcessPoolExecutor() as executor:
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

    # Rerank poses in cases of structures dropped by clashes
    df["rerank_combined"] = (
        df.groupby("pdb_id")["rerank_combined"].rank(ascending=True) - 1
    ).astype(int)
    df["rerank"] = (df.groupby("pdb_id")["rerank"].rank(ascending=True) - 1).astype(int)
    df["rank"] = (df.groupby("pdb_id")["rank"].rank(ascending=True) - 1).astype(int)

    return df


def harmonic_mean(series: list):
    """
    Calculate the harmonic mean of a row-wise operation across multiple pandas Series.

    Parameters:
        series (array-like): A collection of pandas Series or arrays to compute the harmonic mean row-wise.

    Returns:
        pandas.Series or numpy.ndarray: The harmonic mean computed row-wise.
    """
    # Convert inputs to a 2D array
    data = np.vstack(series).T  # Transpose to align for row-wise operations
    if np.any(data <= 0):
        raise ValueError("All values must be positive for harmonic mean.")

    return len(series) / np.sum(1.0 / data, axis=1)


def get_plddts_for_run_dir(run_dir: Path) -> pd.DataFrame:
    """Get pLDDT scores for a run directory.

    Args:
        run_dir (Path): Path to the run directory.

    Returns:
        DataFrame: DataFrame with pLDDT scores.
    """
    # Check if results already exist
    score_file = run_dir / "cdr_peptide_plddt_scores.csv"
    if score_file.exists():
        return pd.read_csv(score_file)

    # Find all ranked PDB files
    pdb_files = sorted(run_dir.glob("ranked*pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No ranked PDB files found in {run_dir}")

    # Process the first PDB file
    model_0 = strucio.load_structure(pdb_files[0])
    tra_seq = pdb_utils.get_sequence_from_chain(model_0, "D")
    trb_seq = pdb_utils.get_sequence_from_chain(model_0, "E")

    # Extract CDR sequences
    cdrs = [
        get_cdr_from_sequence(tra_seq, 1)[0],
        get_cdr_from_sequence(tra_seq, 2)[0],
        get_cdr_from_sequence(tra_seq, 3)[0],
        get_cdr_from_sequence(trb_seq, 1)[0],
        get_cdr_from_sequence(trb_seq, 2)[0],
        get_cdr_from_sequence(trb_seq, 3)[0],
    ]

    # Create DataFrame
    df = pd.DataFrame({"path": [str(p) for p in pdb_files]})
    df[["A1", "A2", "A3", "B1", "B2", "B3"]] = cdrs

    # Compute pLDDT scores
    df = get_plddt_scores(df.copy(deep=True), 1)

    # Save results
    df = df[
        [
            "path",
            "plddt",
            "plddt_include_peptide",
            "plddt_normalized",
            "plddt_include_peptide_normalized",
        ]
    ]
    df.to_csv(score_file, index=False)
    return df


def compute_plddt_sum(
    plddt: np.ndarray,
    indices: list,
    eps: float = 1e-6,
    normalize_on_length: bool = False,
) -> float:
    """Compute summed PLDDT for a CDR.

    Args:
        plddt (np.ndarray): PLDDT scores.
        indices (list): List of lists with CDR indices.
        eps (float): Epsilon value to prevent division by zero.
        normalize_on_length (bool): Normalize on the number of CDR-peptide residues.

    Returns:
        float: Summed PLDDT scores.
    """
    plddt_sum = plddt[indices] + eps

    if normalize_on_length:
        plddt_sum /= sum([len(cdr_idx) for cdr_idx in indices])

    return plddt_sum.item() * 0.01


def get_plddt_score(
    structure: struc.AtomArray,
    a1: str,
    a2: str,
    a3: str,
    b1: str,
    b2: str,
    b3: str,
    top_k: int,
    normalize_on_length: bool = True,
    include_peptide: bool = False,
) -> list:
    """Gets pLDDT scores for CDR123ab-peptide residues.

    Args:
        structure: struc.AtomArray with pLDDT attribute.
        a1: string for CDR1a sequence.
        a2: string for CDR2a sequence.
        a3: string for CDR3a sequence.
        b1: string for CDR1b sequence.
        b2: string for CDR2b sequence.
        b3: string for CDR3b sequence.
        top_k: Number of top models to consider.
        normalize_on_length: Normalize on the number of residues.
        include_peptide: Include peptide residues in the calculation.


    Returns:
        list: List of PLDDT scores.
    """
    if top_k > 1:
        raise NotImplementedError("Top k > 1 not implemented for PLDDT scoring.")

    structure = structure[structure.atom_name == "CA"]

    # Get peptide indices
    if include_peptide:
        peptide_plddt = structure[structure.chain_id == "C"].b_factor
        tra_peptide_indices = [
            np.arange(
                len(structure[structure.chain_id == "D"]),
                len(structure[structure.chain_id == "D"])
                + len(structure[structure.chain_id == "C"]),
            )
        ]
        trb_peptide_indices = [
            np.arange(
                len(structure[structure.chain_id == "E"]),
                len(structure[structure.chain_id == "E"])
                + len(structure[structure.chain_id == "C"]),
            )
        ]
    else:
        peptide_plddt = np.zeros(len(structure[structure.chain_id == "C"]))
        tra_peptide_indices = []
        trb_peptide_indices = []

    # Compute log prob sums for all CDRs
    plddt_sum = compute_plddt_sum(
        plddt=np.concatenate(
            [
                structure[structure.chain_id == "D"].b_factor,
                peptide_plddt,
            ]
        ),
        indices=get_cdr_indices(
            pdb_utils.get_sequence_from_chain(structure, "D"),
            a1,
            a2,
            a3,
        )
        + tra_peptide_indices,
        normalize_on_length=normalize_on_length,
    ) + compute_plddt_sum(
        plddt=np.concatenate(
            [
                structure[structure.chain_id == "E"].b_factor,
                peptide_plddt,
            ]
        ),
        indices=get_cdr_indices(
            pdb_utils.get_sequence_from_chain(structure, "E"),
            b1,
            b2,
            b3,
        )
        + trb_peptide_indices,
        normalize_on_length=normalize_on_length,
    )
    return plddt_sum / 2  # Average over TRA and TRB


def get_plddt_scores(
    df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Get PLDDT scores for a dataframe of TCRs."""
    plddt_normalized = []
    plddt_include_peptide_normalized = []

    for path, a1, a2, a3, b1, b2, b3 in df[
        ["path", "A1", "A2", "A3", "B1", "B2", "B3"]
    ].values:
        structure = strucio.load_structure(path, extra_fields=["b_factor"])
        plddt_normalized.append(
            get_plddt_score(
                structure,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3,
                top_k,
                normalize_on_length=True,
                include_peptide=False,
            )
        )
        plddt_include_peptide_normalized.append(
            get_plddt_score(
                structure,
                a1,
                a2,
                a3,
                b1,
                b2,
                b3,
                top_k,
                normalize_on_length=True,
                include_peptide=True,
            )
        )
    df["plddt_normalized"] = plddt_normalized
    df["plddt_include_peptide_normalized"] = plddt_include_peptide_normalized
    return df[
        [
            "path",
            "plddt_normalized",
            "plddt_include_peptide_normalized",
        ]
    ]
