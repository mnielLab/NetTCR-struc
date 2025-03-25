from pathlib import Path

import pandas as pd
from tqdm import tqdm


def create_df_from_dir(input_dir: Path) -> pd.DataFrame:
    """Create a dataframe from a directory of structure modeling runs.

    Args:
        input_dir: Path to input directory.

    Returns:
        A dataframe with columns name and path.
    """
    rows = []

    # Iterate over each subdirectory in the input directory
    for sub_dir in tqdm(input_dir.iterdir()):
        if sub_dir.is_dir():
            # Glob pdb files within the current subdirectory
            pdb_files = list(sub_dir.glob("ranked*.pdb")) + list(
                sub_dir.glob("ranked*.cif")
            )
            for pdb_file in pdb_files:
                name = f"{pdb_file.parent.name}_ranked_{pdb_file.stem.split('_')[-1]}"
                rows.append((name, pdb_file))

    return pd.DataFrame(rows, columns=["name", "path"])
