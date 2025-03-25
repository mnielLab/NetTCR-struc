import argparse
import concurrent.futures
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from nettcrstruc.dataset.processing import ProteinGraphDatasetPreprocessor
from nettcrstruc.utils import pdb_utils
from nettcrstruc.utils.pdb_utils import get_sequence_from_chain
from nettcrstruc.utils.utils import create_df_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract geometric features from protein structures and save as graphs."
    )
    parser.add_argument(
        "-i",
        dest="input",
        type=Path,
        help="Path to input csv file.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to input directory.",
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
        "-f",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "-d",
        dest="device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "-p",
        dest="config_path",
        type=Path,
        help="Path to config file.",
        default=None,
    )
    parser.add_argument(
        "-e",
        dest="esm_if1_feature_dir",
        type=Path,
        help="Path to esm if1 features.",
        default=None,
    )
    return parser.parse_args()


def get_esm_if1_features(
    feature_path: Path,
    chain_id,
    structure,
) -> np.ndarray:
    features = torch.load(feature_path, map_location=torch.device("cpu"))

    _, idx = np.unique(chain_id, return_index=True)
    encodings = []
    for chain in chain_id[np.sort(idx)]:
        sequence = get_sequence_from_chain(structure, chain)
        # Get the index of the chain in the esm_if1_features
        chain_idx = np.where(np.array(features["sequence_order"]) == sequence)[0][0]

        encodings.append(features["encoding"][chain_idx].cpu().detach())

    return {
        "residue_if_encodings": np.concatenate(encodings, axis=0),
    }


def process_row_as_residue_graph(
    row: list,
    out_dir: Path,
    overwrite: bool,
    device: str,
    esm_if1_feature_dir: Path,
    feature_config: dict,
) -> None:
    """Process and save a row of the dataframe.

    Args:
        row: list with name, pdb_path, dockq, fnat, fnonnat, lmrs, and irms.
        out_dir: Output directory to save the processed file.
        overwrite: Flag to overwrite existing files.
        device: Computing device to use.
        esm_if1_feature_dir: Path to ESM-IF1 features.
        feature_config: Dict with which extra features to include.
    """
    if len(row) > 2:
        name, pdb_path, dockq, fnat, fnonnat, lrms, irms = row
    else:
        name, pdb_path = row
        dockq, fnat, fnonnat, lrms, irms = None, None, None, None, None
    output_file = out_dir / f"{name}.pt"

    # Skip processing if file exists and overwrite is False
    if output_file.exists() and not overwrite:
        return

    data_processor = ProteinGraphDatasetPreprocessor(
        top_k=30,
        num_rbf=16,
        device=device,
    )

    # Fetch structure data
    structure = strucio.load_structure(pdb_path, extra_fields=["b_factor"])

    # Merge MHC class II B chain to A
    structure = pdb_utils.merge_mhc_chains(structure)

    # Set order of chains to always be the same
    structure = pdb_utils.set_chain_order(structure, ["D", "E", "C", "A"])

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

    # Add extra features
    if esm_if1_feature_dir:
        feature_dict = get_esm_if1_features(
            esm_if1_feature_dir / f"{name}.pt",
            structure.chain_id,
            structure,
        )
    else:
        feature_dict = {}

    # Slice extra features to indices
    for key, value in feature_dict.items():
        if value is not None:
            feature_dict[key] = value

    # Construct extra features
    extra_node_scalar_features = [
        feat
        for name, feat in feature_dict.items()
        if name in feature_config["node_scalar_features"]
    ]
    extra_node_vector_features = [
        feat
        for name, feat in feature_dict.items()
        if name in feature_config["node_vector_features"]
    ]
    extra_edge_scalar_features = [
        feat
        for name, feat in feature_dict.items()
        if name in feature_config["edge_scalar_features"]
    ]
    extra_edge_vector_features = [
        feat
        for name, feat in feature_dict.items()
        if name in feature_config["edge_vector_features"]
    ]

    # Featurize structure as graph
    data = data_processor._featurize_as_graph(
        name=name,
        seq=sequence,
        coords=backbone_coords,
        extra_node_scalar_features=extra_node_scalar_features,
        extra_node_vector_features=extra_node_vector_features,
        extra_edge_scalar_features=extra_edge_scalar_features,
        extra_edge_vector_features=extra_edge_vector_features,
        chain_id=chain_id,
    )

    # Add target values
    if dockq is not None:
        data.dockq = dockq
        data.fnat = fnat
        data.fnonnat = fnonnat
        data.lrms = lrms
        data.irms = irms

    # Saving the processed data
    data = data.to("cpu")
    torch.save(data, output_file)


def process_batch(
    batch: list,
    out_dir: Path,
    overwrite: bool,
    device: str,
    esm_if1_feature_dir: Path,
    feature_config: dict,
) -> None:
    """Process a batch of rows.

    Args:
        batch: A list of rows to process.
        out_dir: Output directory path.
        overwrite: Flag to overwrite existing files.
        device: Computing device to use.
        esm_if1_feature_dir: Path to ESM-IF1 features.
        feature_config: Dict with which extra features to include.

    Returns:
        A list of results from processing each row in the batch.
    """
    results = []
    for row in batch:
        # Process each row here and append the result
        result = process_row_as_residue_graph(
            row=row,
            out_dir=out_dir,
            overwrite=overwrite,
            device=device,
            esm_if1_feature_dir=esm_if1_feature_dir,
            feature_config=feature_config,
        )
        results.append(result)
    return results


def main() -> None:
    """Main entry point for GVP data preprocessing."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = create_df_from_dir(args.input_dir)
    rows = df[["name", "path"]].values

    # Splitting rows into batches
    batches = np.array_split(rows, args.num_workers)

    # Load config with features to use
    if args.config_path and args.config_path.exists():
        with open(args.config_path, "r") as f:
            feature_config = yaml.safe_load(f)
    else:
        feature_config = {
            "node_scalar_features": [],
            "node_vector_features": [],
            "edge_scalar_features": [],
            "edge_vector_features": [],
        }

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        # Map each batch to a process
        future_to_batch = {
            executor.submit(
                process_batch,
                batch,
                args.out_dir,
                args.overwrite,
                args.device,
                args.esm_if1_feature_dir,
                feature_config,
            ): batch
            for batch in batches
        }

        # Collecting results using tqdm for progress tracking
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch), total=len(batches)
        ):
            results.extend(future.result())


if __name__ == "__main__":
    main()
