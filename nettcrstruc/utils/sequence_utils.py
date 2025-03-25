from typing import Union

import numpy as np
from anarci import number


def get_cdr_from_sequence(sequence: str, cdr: Union[str, int]) -> tuple:
    """Returns sequence and indices of specified CDR in input sequence.

    Args:
        sequence (str): Input sequence.
        cdr (str or int): CDR to extract. Can be 1, 2, 3.

    Returns:
        tuple: CDR sequence and indices.
    """
    if isinstance(cdr, str):
        cdr = int(cdr)

    anarci_out = number(
        sequence,
        scheme="imgt",
    )

    # Get CDR sequence
    cdr_sequence = []

    # Set indices based on CDR
    if cdr == 1:
        start = 26
        end = 39
    elif cdr == 2:
        start = 55
        end = 66
    elif cdr == 3:
        start = 104
        end = 118

    for pos in anarci_out[0]:
        if start <= pos[0][0] <= end:
            cdr_sequence.append(pos[1])
    cdr_sequence = "".join(cdr_sequence)
    cdr_sequence = cdr_sequence.replace("-", "")

    # Find indices of CDR in input (anarci might change the length of the sequence)
    cdr_start_idx = sequence.find(cdr_sequence)
    cdr_indices = list(range(cdr_start_idx, cdr_start_idx + len(cdr_sequence)))

    return cdr_sequence, cdr_indices


def get_cdr_indices(sequence: str, cdr1: str, cdr2: str, cdr3: str) -> list:
    """Get indices of CDRs in a sequence.

    Args:
        sequence (str): Amino acid sequence.
        cdr1 (str): CDR1 sequence.
        cdr2 (str): CDR2 sequence.
        cdr3 (str): CDR3 sequence.

    Returns:
        list: List of indices for CDRs.
    """
    indices = []
    for cdr in [cdr1, cdr2, cdr3]:
        start_idx = sequence.find(cdr)
        end_idx = start_idx + len(cdr)
        indices.append(np.arange(start_idx, end_idx))
    return indices
