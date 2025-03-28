from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_paths_from_dir(input_dir: Path) -> list:
    """Create a list of paths from a directory of structure modeling runs.

    Args:
        input_dir: Path to input directory.

    Returns:
        A list of paths.
    """
    paths = []

    # Iterate over each subdirectory in the input directory
    for sub_dir in tqdm(input_dir.iterdir()):
        if sub_dir.is_dir():
            # Glob pdb files within the current subdirectory
            files = list(sub_dir.glob("*.pdb")) + list(sub_dir.glob("*.cif"))
            for path in files:
                paths.append(path)
    return paths
