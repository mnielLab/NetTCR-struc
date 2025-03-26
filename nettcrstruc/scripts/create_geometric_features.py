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
from nettcrstruc.utils.utils import create_df_from_dir


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
    return parser.parse_args()


def process_entry(
    pdb_path: Path,
    out_dir: Path,
    name: str,
    device: str,
    esm_if1_model: Any,
    alphabet: Any,
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
    sequence, chain_id, backbone_coords, structure = extract_features_from_pdb(pdb_path)

    # Skip processing if file exists and overwrite is False
    esm_f1_features = get_esm_if1_features(
        structure=structure,
        pdb_path=pdb_path,
        chain_id=chain_id,
        feature_path=out_dir / "esm_if1_embeddings" / f"{name}.pt",
        esm_if1_model=esm_if1_model,
        alphabet=alphabet,
        device=device,
    )
    geoemtric_features = get_geometric_features(
        feature_path=out_dir / "gvp" / f"{name}.pt",
        name=name,
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
    torch.save(geoemtric_features, out_dir / "gvp_if1_embeddings" / f"{name}.pt")


def process_batch(
    batch: list,
    out_dir: Path,
    device: str,
) -> str:
    """Process a batch of entries.

    Args:
        batch: A list of tuples with the name and path to the PDB file.
        out_dir: The output directory.
        device: The device to use for computation.

    Returns:
        A list of results from processing each row in the batch.
    """
    esm_if1_model, alphabet = initialize_esm_if1_model(device)

    results = []
    for name, pdb_path in tqdm(batch):
        result = process_entry(
            pdb_path=pdb_path,
            name=name,
            out_dir=out_dir,
            device=device,
            esm_if1_model=esm_if1_model,
            alphabet=alphabet,
        )
        results.append(result)


def main() -> None:
    """Main entry point for GVP data preprocessing."""
    args = parse_args()
    (args.out_dir / "esm_if1_embeddings").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gvp").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gvp_if1_embeddings").mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        df = create_df_from_dir(args.input_dir)
    rows = df[["name", "path"]].values

    batches = np.array_split(rows, args.num_workers)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        future_to_batch = {
            executor.submit(
                process_batch,
                batch,
                args.out_dir,
                args.device,
            ): batch
            for batch in batches
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_batch):
            results.extend(future.result())


if __name__ == "__main__":
    main()
