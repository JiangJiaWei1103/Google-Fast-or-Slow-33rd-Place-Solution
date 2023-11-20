"""Dump node_config_feat of layout collection separately.

The file structure is shown as follows:
./data/processed/layout/
    ├── nlp/
        ├── random/
            ├── node_config_feat/
                ├── <file_name>.npz
Each .npz contains only `node_config_feat` field.
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from metadata import SPLIT

LAYOUT_RAW_ROOT = Path("./data/raw/npz_all/npz/layout")
LAYOUT_PROC_ROOT = Path("./data/processed/layout")
SRC = ["nlp", "xla"]
SEARCH = ["random", "default"]


def _dump_node_config_feat(src: str, search: str) -> None:
    data_root = LAYOUT_RAW_ROOT / f"{src}/{search}"
    for split in SPLIT:
        if src == "nlp" and search == "random" and split == "train":
            continue
        print(f"{split}...")
        split_root = data_root / split
        print(split_root)
        dump_root = LAYOUT_PROC_ROOT / f"{src}/{search}/node_config_feat/{split}"

        for data_file in tqdm(sorted(split_root.glob("*.npz"))):
            file_name = str(data_file).split("/")[-1]
            data_tmp = dict(np.load(data_file))
            node_config_feat_tmp = data_tmp["node_config_feat"]
            np.savez_compressed(dump_root / file_name, node_config_feat=node_config_feat_tmp)


def main() -> None:
    for src in SRC:
        for search in SEARCH:
            print(f"== {src} - {search} ==")
            _dump_node_config_feat(src, search)


if __name__ == "__main__":
    main()
