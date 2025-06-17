import argparse
import concurrent.futures
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nettcrstruc.dataset.esm_if1_utils import (
    get_esm_if1_features,
    initialize_esm_if1_model,
)
from nettcrstruc.dataset.processing import (
    extract_features_from_pdb,
    get_geometric_features,
)
from nettcrstruc.utils.utils import get_paths_from_dir
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract geometric features from protein structures and save as graphs."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        help="Path to input directory.",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        help="Path to input csv file (alternative to providing a directory).",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        type=Path,
        help="Path to output directory.",
        required=True,
    )
    parser.add_argument(
        "-n",
        dest="num_workers",
        type=int,
        help="Number of processes to use.",
        default=1,
    )
    parser.add_argument(
        "-d",
        dest="device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--chain_names",
        type=str,
        default=["D", "E", "C", "A", "B"],
        nargs="+",
    )
    return parser.parse_args()


def mk_feature_dir(pdb_path: Path, out_dir: Path) -> Path:
    """Create a feature dir for a modeling run dir.

    Args:
        pdb_path: Path to the PDB file.
        out_dir: The output directory.

    Returns:
        Path to the feature directory.
    """
    out_dir.mkdir(exist_ok=True)
    feature_dir = out_dir / pdb_path.parent.stem
    feature_dir.mkdir(exist_ok=True)
    return feature_dir


def process_entry(
    pdb_path: Path,
    out_dir: Path,
    device: str,
    esm_if1_model: Any,
    alphabet: Any,
    chain_names: list,
):
    """Process a single entry.

    Args:
        pdb_path: Path to the PDB file.
        out_dir: The output directory.
        name: The name of the entry.
        device: The device to use for computation.
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet for the ESM-IF1 model.
    """
    esm_if1_embeddings_dir = mk_feature_dir(pdb_path, out_dir / "esm_if1_embeddings")
    gvp_dir = mk_feature_dir(pdb_path, out_dir / "gvp")
    gvp_if1_dir = mk_feature_dir(pdb_path, out_dir / "gvp_if1_embeddings")
    
    sequence, chain_id, backbone_coords, structure = extract_features_from_pdb(
        pdb_path,
        chain_names,
    )
    
    file_name = f"{Path(pdb_path.stem)}.pt"
    esm_f1_features = get_esm_if1_features(
        structure=structure,
        pdb_path=pdb_path,
        chain_id=chain_id,
        feature_path=esm_if1_embeddings_dir / file_name,
        esm_if1_model=esm_if1_model,
        alphabet=alphabet,
        device=device,
    )
    geoemtric_features = get_geometric_features(
        feature_path=gvp_dir / file_name,
        seq=sequence,
        coords=backbone_coords,
        chain_id=chain_id,
        device=device,
    )
    geoemtric_features.node_s = torch.cat(
        (
            geoemtric_features.node_s,
            esm_f1_features,
        ),
        dim=-1,
    )
    torch.save(geoemtric_features, gvp_if1_dir / file_name)


def process_batch(
    batch: list,
    out_dir: Path,
    device: str,
    chain_names: list,
) -> str:
    """Process a batch of entries.

    Args:
        batch: A list of tuples with the name and path to the PDB file.
        out_dir: The output directory.
        device: The device to use for computation.
        chain_names: list of chain identifiers, in the order TCRa, TCRb, peptide, MHCa, MHCb.

    Returns:
        A list of results from processing each row in the batch.
    """
    esm_if1_model, alphabet = initialize_esm_if1_model(device)

    results = []
    for pdb_path in tqdm(batch):
        result = process_entry(
            pdb_path=Path(pdb_path),
            out_dir=out_dir,
            device=device,
            esm_if1_model=esm_if1_model,
            alphabet=alphabet,
            chain_names=chain_names,
        )
        results.append(result)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        paths = pd.read_csv(args.input_csv)["path"].values
    else:
        paths = get_paths_from_dir(args.input_dir)

    batches = np.array_split(paths, args.num_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_batch,
                batch,
                args.out_dir,
                args.device,
                args.chain_names,
            )
            for batch in batches
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
