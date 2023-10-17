"""
Project metadata for global access.
Author: JiaWei
"""
# ==Data==
# Compiler optimization
OPTIM = ["layout", "tile"]
# Source
SRC = ["xla", "nlp"]
# Search strategy
SEARCH = ["default", "random"]
# Dataset split
SPLIT = ["train", "valid", "test"]
# Collection
COLL = ["layout-nlp-default", "layout-nlp-random", "layout-xla-default", "layout-xla-random", "tile-xla"]

# ====
# Number of unique op-codes
NODE_FEAT_DIM = 86  # 101 #140
CONFIG_FEAT_DIM = 24
NODE_CONFIG_FEAT_DIM = 18
N_OPS = 120
