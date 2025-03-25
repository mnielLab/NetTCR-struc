import argparse
import concurrent.futures
from pathlib import Path
from typing import Any, List, Tuple, Union

import esm
import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser, PDBParser
from tqdm import tqdm

from nettcrstruc.utils.utils import create_df_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--out_dir",
        type=Path,
        help="Path to output directory",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use.",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_known_args()[0]


def read_pdb_structure(
    pdb_file: Union[Path, str],
    pdb_id: str = "foo",
    modelnr: int = 0,
    return_all_models: bool = False,
) -> Any:
    """
    Reads the structure from a PDB file.

    Args:
        pdb_file: A Pathlib object or string path to the PDB file.
        pdb_id: The ID of the PDB structure.
        modelnr: The model number to be read from the PDB file.
        return_all_models: If True, returns all models from the PDB file.

    Returns:
        The structure of the specified model from the PDB file.
    """
    # reading model 0 by default
    assert isinstance(
        modelnr, int
    ), f"Model number needs to be a valid integer, it was {modelnr}"

    # Check if filetype is pdb or cif
    pdb_file = Path(pdb_file)
    if pdb_file.suffix == ".cif":
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)

    # return all models
    if return_all_models:
        models = list()
        for m in structure:
            models.append(m)
        return models

    # return only desired model
    else:
        return structure[modelnr]


def compute_esm_if1_on_pdb(
    pdb_file: Union[Path, str],
    esm_if1_model: Any,
    alphabet: Any,
) -> Tuple[List[Any], List[Any], List[str]]:
    """
    Compute ESM-IF1 encodings and probabilities on a PDB file using esm-if1 multichain.

    Args:
        pdb_file: A Pathlib object or string path to the PDB file.
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet parameter from the ESM-IF1 model.
        esmif1_enc: If True, computes ESM-IF1 encodings.
        esmif1_prob: If True, computes ESM-IF1 probabilities.

    Returns:
        A tuple containing lists of tensor encodings, probabilities, and corresponding sequences for the PDB.
    """
    structure = read_pdb_structure(pdb_file)
    esmif1_encs, esmif1_probs, sequence_order, chain_order = [], [], [], []
    chain_names = [c.get_id() for c in structure.get_chains()]

    # load pdb structure into coordinates
    esm_if1_loaded_structure = esm.inverse_folding.util.load_structure(
        str(pdb_file), chain_names
    )
    (
        coords,
        native_seqs,
    ) = esm.inverse_folding.multichain_util.extract_coords_from_complex(
        esm_if1_loaded_structure
    )
    # extract esm_if1 encodings and probabilities
    for c in chain_names:
        seq = native_seqs[c]
        esmif1_e, esmif1_p = forward_pass(
            esm_if1_model,
            alphabet,
            coords,
            c,
            seq,
        )

        # append encodings, probalities and sequence string.
        esmif1_encs.append(esmif1_e)
        esmif1_probs.append(esmif1_p)
        sequence_order.append(seq)
        chain_order.append(c)

    return esmif1_encs, esmif1_probs, sequence_order, chain_order


def forward_pass(
    model: Any,
    alphabet: Any,
    coords: Any,
    chain: str,
    seq: str,
):
    """
    Compute ESM-IF1 encodings and decoder probabilities for a specific chain in a PDB file.

    Args:
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet parameter from the ESM-IF1 model.
        coords: Coordinates of the chain.
        chain: The chain identifier.
        seq: Sequence of the chain.

    Returns:
        A tuple containing the ESM-IF1 encoding and probability for the given chain.
    """
    with torch.no_grad():
        # Preprocess coordinates
        device = next(model.parameters()).device
        all_coords = esm.inverse_folding.multichain_util._concatenate_coords(
            coords, chain
        )
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
        batch = [(all_coords, None, seq)]
        batch_coords, confidence, _, tokens, padding_mask = batch_converter(
            batch, device=device
        )

        # Forward pass
        encoder_out = model.encoder.forward(
            batch_coords,
            padding_mask,
            confidence,
            return_all_hiddens=False,
        )
        logits, _ = model.decoder(
            tokens[:, :-1].to(device),
            encoder_out=encoder_out,
            features_only=False,
            return_all_hiddens=False,
        )

        # Prepare encoder outputs for saving
        target_chain_len = coords[chain].shape[0]
        encoder_out = encoder_out["encoder_out"][0][1:-1, 0]
        encoder_out = encoder_out[:target_chain_len].cpu().detach()

        # Prepare decoder outputs for saving
        mm = torch.nn.Softmax(dim=1)
        logits_permute = torch.squeeze(torch.permute(logits, (0, 2, 1)))

        all_probs = mm(logits_permute).cpu().detach()
        tgt_idxs = torch.squeeze(tokens[:, 1:]).cpu().detach()
        res_probs = torch.stack(
            [all_probs[i, tgt_idxs[i].item()] for i in range(len(tgt_idxs))]
        )
        return encoder_out, res_probs


def process_batch(batch: list, out_dir: Path, device: str, overwrite: bool) -> None:
    """Process a batch of rows.

    Args:
        batch: A list of rows to process.
        out_dir: Output directory path.
        overwrite: Flag to overwrite existing files.
        device: Computing device to use.

    Returns:
        A list of results from processing each row in the batch.
    """
    esm_if1_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_if1_model = esm_if1_model.eval()
    esm_if1_model.to(device)
    for name, pdb_file in tqdm(batch):
        output_file = out_dir / f"{name}.pt"
        if output_file.exists() and not overwrite:
            continue
        # Process each row here and append the result
        esmif1_encs, esmif1_probs, sequence_order, chain_order = compute_esm_if1_on_pdb(
            pdb_file,
            esm_if1_model,
            alphabet,
        )
        data = {
            "encoding": esmif1_encs,
            "probabilities": esmif1_probs,
            "sequence_order": sequence_order,
            "chain_order": chain_order,
        }
        torch.save(data, output_file)
    return "Done"


def main() -> None:
    """Main entry point for extracting ESM-IF1 embeddings from PDB files."""
    torch.multiprocessing.set_start_method("spawn")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = create_df_from_dir(args.input_dir)
    rows = df[["name", "path"]].values

    batches = np.array_split(rows, args.num_workers)

    # Submit each batch to job queue
    if args.num_workers == 1:
        for batch in batches:
            process_batch(
                batch,
                args.out_dir,
                torch.device("cuda") if args.device == "gpu" else torch.device("cpu"),
                args.overwrite,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            # Map each batch to a process
            future_to_batch = {
                executor.submit(
                    process_batch,
                    batch,
                    args.out_dir,
                    (
                        torch.device("cuda")
                        if (args.device == "gpu" or args.device == "cuda")
                        else torch.device("cpu")
                    ),
                    args.overwrite,
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
