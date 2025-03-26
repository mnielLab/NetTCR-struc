from pathlib import Path
from typing import Any, List, Tuple, Union

import biotite.structure as struc
import esm
import numpy as np
import torch

from nettcrstruc.utils.pdb_utils import get_sequence_from_chain


def compute_esm_if1_on_pdb(
    pdb_path: Union[Path, str],
    esm_if1_model: Any,
    alphabet: Any,
    chain_names: List[str],
) -> Tuple[List[Any], List[Any], List[str]]:
    """
    Compute ESM-IF1 encodings and probabilities on a PDB file using esm-if1 multichain.

    Args:
        pdb_path: A Pathlib object or string path to the PDB file.
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet for the ESM-IF1 model.
        chain_names: List of chain names in the PDB file.

    Returns:
        A tuple containing lists of tensor encodings, probabilities, and corresponding sequences for the PDB.
    """
    esmif1_encs, esmif1_probs, sequence_order, chain_order = [], [], [], []

    # load pdb structure into coordinates
    esm_if1_loaded_structure = esm.inverse_folding.util.load_structure(
        str(pdb_path), chain_names
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
            model=esm_if1_model,
            alphabet=alphabet,
            coords=coords,
            chain=c,
            seq=seq,
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
        model: The ESM-IF1 model used for computation.
        alphabet: Alphabet for the ESM-IF1 model.
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


def initialize_esm_if1_model(device) -> Tuple[Any, Any]:
    """
    Initialize the ESM-IF1 model and alphabet.

    Args:
        device: The device to use for computation.
    """
    esm_if1_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_if1_model = esm_if1_model.eval()
    esm_if1_model.to(device)
    return esm_if1_model, alphabet


def generate_esm_f1_features(
    pdb_path: Path,
    output_path: Path,
    esm_if1_model: Any,
    alphabet: Any,
    chain_names: List[str],
) -> dict:
    """Extracts structural embeddings from ESM-IF1.

    Args:
        pdb_path: Path to the PDB file.
        output_path: Path to save the extracted features.
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet for the ESM-IF1 model.
        chain_names: List of chain names in the PDB file.

    Returns:
        A dict with ESM-IF1 features for a structure.
    """
    # Process each row here and append the result
    esmif1_encs, esmif1_probs, sequence_order, chain_order = compute_esm_if1_on_pdb(
        pdb_path=pdb_path,
        esm_if1_model=esm_if1_model,
        alphabet=alphabet,
        chain_names=chain_names,
    )
    data = {
        "encoding": esmif1_encs,
        "probabilities": esmif1_probs,
        "sequence_order": sequence_order,
        "chain_order": chain_order,
    }
    torch.save(data, output_path)
    return data


def get_esm_if1_features(
    feature_path: Path,
    pdb_path: Path,
    chain_id: np.ndarray,
    structure: struc.AtomArray,
    esm_if1_model: Any,
    alphabet: Any,
    device: str,
) -> np.ndarray:
    """Helper function for extracting or loading ESM-IF1 features.

    Args:
        feature_path: Path to save/load features.
        pdb_path: Path to the PDB file.
        chain_id: Chain ID for each residue.
        structure: struc.AtomArray structure.
        esm_if1_model: The ESM-IF1 model used for computation.
        alphabet: Alphabet for the ESM-IF1 model.
        device: If "cuda", preprocessing is performed on the GPU.

    Returns:
        A tensor of ESM-IF1 representations.
    """
    _, idx = np.unique(chain_id, return_index=True)
    unique_chain_id = chain_id[np.sort(idx)]
    if feature_path.exists():
        features = torch.load(feature_path, map_location=torch.device(device))
    else:
        features = generate_esm_f1_features(
            pdb_path=pdb_path,
            output_path=feature_path,
            chain_names=unique_chain_id.tolist(),
            esm_if1_model=esm_if1_model,
            alphabet=alphabet,
        )

    encodings = []
    for chain in unique_chain_id:
        sequence = get_sequence_from_chain(structure, chain)
        # Get the index of the chain in the esm_if1_features
        chain_idx = np.where(np.array(features["sequence_order"]) == sequence)[0][0]

        encodings.append(features["encoding"][chain_idx].cpu().detach())

    return torch.from_numpy(np.concatenate(encodings, axis=0)).to(device)
