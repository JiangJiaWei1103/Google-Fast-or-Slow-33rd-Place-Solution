"""
Generate layout DataFrame.
Author: JiaWei Jiang
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from metadata import SPLIT
from paths import PROC_DATA_PATH, RAW_DATA_PATH

# 4 combinations in total
coll = "xla-default"
src, search = coll.split("-")
DATA_ROOT = Path(RAW_DATA_PATH) / f"npz_all/npz/layout/{src}/{search}"
DUMP_DIR = Path(PROC_DATA_PATH) / f"layout/{src}/{search}"


def main() -> None:
    """Generate layout DataFrame for different splits.

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
            layout_tmp = dict(np.load(data_file))
            layout_tmp["file"] = file
            layout_tmp.pop("node_config_feat")  # Too large to handle now
            df_split.append(layout_tmp)

        df_split = pd.DataFrame.from_dict(df_split)
        df_split["split"] = split
        df_split.to_pickle(DUMP_DIR / f"{split}.pkl")


if __name__ == "__main__":
    main()
