# MICROUNET — Finding Generalizable Edge U-Nets

**BIMAP SS26 | FAU Erlangen-Nürnberg**

## Project Goal
Investigate the relationship between model size and performance for U-Net 
architectures under 0.1M parameters across biomedical segmentation datasets.

## Phase 0 Results
- Model: MicroUNet (4 layers, 8 filters, SeparableConv2D, BatchNorm)
- Parameters: 63,628 (limit: <100,000)
- Dataset: BAGLS (laryngeal endoscopy, test split, 3500 images)
- Val IoU: 0.8058 ± 0.0121 (runs 001-003, seeds 42/43/44)

## Setup
```bash
source ~/.venvs/torch312/bin/activate
```

## Run Training
```bash
python3 train.py
```

## Project Structure
microunet/
├── data/
│   ├── dataset.py      # BAGLS dataset loader
│   └── test/           # BAGLS test images + masks
├── models/
│   └── unet.py         # MicroUNet architecture
├── experiments/        # config.yaml per run
├── experiments.csv     # experiment log
├── train.py            # training script
└── README.md

## Experiment Log
See `experiments.csv` for all runs.
Each run has a config in `experiments/config_XXX.yaml`.

## Reference
Kist & Döllinger, IEEE Access 2020