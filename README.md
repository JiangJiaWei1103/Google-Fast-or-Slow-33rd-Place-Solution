# Google - Fast or Slow? Predict AI Model Runtime - 33rd Place Solution

> Solution writeup: [33rd Solution Writeup and Discussion on GST](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456579)

## How to Run
### 1. Download Dataset
You have to follow instructions on the [data tab](https://www.kaggle.com/competitions/predict-ai-model-runtime/data) to download the dataset. Then, you need to unzip and put the raw data under the directory `./data/raw/`. Under `./data/raw/`, the folder structure looks like the follows,
```console
npz_all/npz/
├── layout/
│   ├── nlp/
│   │   ├── default/
│   │   │   ├── test/
│   │   │   ├── train/
│   │   │   └── valid/
│   │   └── random/
│   │       ├── test/
│   │       ├── train/
│   │       └── valid/
│   └── xla/
│       ├── default/
│       │   ├── test/
│       │   ├── train/
│       │   └── valid/
│       └── random/
│           ├── test/
│           ├── train/
│           └── valid/
└── tile/
    └── xla/
        ├── test/
        ├── train/
        └── valid/
```

### 2. Prepare Data
To obtain the processed data, please run the following commands,
```
python -m data.preparation.gen_tile_df
python -m data.preparation.gen_layout_df
python -m data.preparation.dump_layout_node_config_feat
```
As can be observed in the folder structure of `./data/raw/`, there are five data collections in total (*e.g.,* *layout-nlp-default*), each of which has its own **processed** counterpart. Let's take *layout-nlp-default* as an example. Under `./data/processed/`, the folder structure is shown as follows,
```console
layout/nlp/default/
├── train.pkl
├── valid.pkl
├── test.pkl
└── node_config_feat/
    ├── test/
    ├── train/
    └── valid/
```

### 3. Train Models
Each experiment is controlled via 3 configuration files (somewhat annoying, which I'll improve in the next project), `./config/defaults.yaml`, `./config/dp.yaml` and `./config/<model_name>.yaml`, where `<mode_name>` is the name of the model architecture. After setup, you can now train models by running,
```
# Note: If you don't want to track the experiment with wandb, please set it to False

# Train tile model
python -m tools.main --project-name <any_project_name> --model-name <model_name> --use-wandb True

# Train layout model
python -m tools.main_layout --project-name <any_project_name> --model-name <model_name> --use-wandb True
```
The output objects (*e.g.,* model checkpoint `model.pth`, log file `train_eval.log`) will be dumped under the path `./output/<%m%d-%H_%M_%S>/`, because we use timestamp to do a simple version control.

### 4. Run Inference
After models are trained, you can run inference on test set for the final submission,
```
# Run inference on tile
python -m tools.infer --exp-id <%m%d-%H_%M_%S> --data-split test --model-name <model_name> --mid last --seeds 0

# Infer layout
python -m tools.infer_layout --exp-id <%m%d-%H_%M_%S> --coll <collection> --data-split test --model-name <model_name> --mid <model_identifier> --seeds <seeds>

# options:
# --exp-id            experiment identifier represented as timestamp
# --coll              {xla-default, xla-random, nlp-default, nlp-random}
# --mid               model checkpoint identifier, always choose last to avoid overfitting on validation set
# --seeds             [0, 1, ...], feel free to ensemble with models trained on different random seeds
```

### 5. Experimental Results
The local CV score of my final best result is shown in the following table,
| Collection | CV | 
| --- | --- | 
| *tile* | 0.9551 | 
| *xla-default* | 0.3188 |
| *xla-random* | 0.5569 |
| *nlp-default* | 0.5053 |
| *nlp-random* | 0.8845 |

The average score across all data collections are shown as follows,
|  | Average Score | 
| --- | --- | 
| CV | 0.64412 | 
| Public LB | 0.65282 | 
| Private LB | 0.62695 | 

