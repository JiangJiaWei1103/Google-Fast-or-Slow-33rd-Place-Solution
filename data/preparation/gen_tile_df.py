"""
Generate tile DataFrame.
Author: JiaWei Jiang
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from metadata import SPLIT
from paths import PROC_DATA_PATH, RAW_DATA_PATH

COLL = "tile-xla"
DATA_ROOT = Path(RAW_DATA_PATH) / "npz_all/npz/tile/xla"
DUMP_DIR = Path(PROC_DATA_PATH) / "tile/xla"


def main() -> None:
    """Generate tile DataFrame for different splits.

    Return:
        None
    """
    for split in SPLIT:
        print(f"Generate tile DataFrame for {split}...")
        split_root = DATA_ROOT / split

        df_split = []
        for file in tqdm(os.listdir(split_root)):
            if not file.endswith(".npz"):
                continue

            data_file = split_root / file
            tile_tmp = dict(np.load(data_file))
            tile_tmp["file"] = file
            df_split.append(tile_tmp)

        df_split = pd.DataFrame.from_dict(df_split)
        df_split["split"] = split
        df_split.to_pickle(DUMP_DIR / f"{split}.pkl")


if __name__ == "__main__":
    main()
