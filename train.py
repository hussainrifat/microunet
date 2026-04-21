# train.py
# Training script for MicroUNet on BAGLS dataset
# Run: python3 train.py

import os
import csv
import yaml
import torch
import torch.nn as nn
from datetime import date
from data.dataset import get_dataloaders
from models.unet import MicroUNet, count_parameters


# ── IoU metric ──────────────────────────────────────────────────────────────

def compute_iou(preds, masks, threshold=0.5):
    """
    Computes Intersection over Union (IoU) for binary segmentation.
    
    IoU = (Area of Overlap) / (Area of Union)
    IoU = 1.0 means perfect prediction, 0.0 means no overlap at all.
    
    preds: model output, shape [B, 1, H, W], values 0.0-1.0
    masks: ground truth, shape [B, 1, H, W], values 0 or 1
    threshold: anything above 0.5 is predicted as foreground
    """
    # Binarize predictions: 0.7 -> 1, 0.3 -> 0
    preds_binary = (preds > threshold).float()

    # Intersection: pixels that are foreground in BOTH prediction and ground truth
    intersection = (preds_binary * masks).sum(dim=(1, 2, 3))

    # Union: pixels that are foreground in EITHER prediction or ground truth
    union = (preds_binary + masks).clamp(0, 1).sum(dim=(1, 2, 3))

    # Avoid division by zero: if union is 0, both pred and mask are empty -> IoU=1
    iou = torch.where(union == 0, torch.ones_like(intersection), intersection / union)

    return iou.mean().item()  # return average IoU across the batch


# ── Experiment logging ───────────────────────────────────────────────────────

def save_config(config, run_id):
    """Saves experiment configuration as a yaml file."""
    os.makedirs("experiments", exist_ok=True)
    config_path = f"experiments/config_{run_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_path}")


def log_experiment(run_id, config, val_iou):
    """
    Appends one row to experiments.csv.
    This is the mandatory experiment log the professor requires.
    """
    log_path = "experiments.csv"
    # Check if file is empty to write header
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "date", "model", "dataset", "seed",
                "epochs", "batch_size", "lr", "base_filters", "val_iou"
            ])
        writer.writerow([
            run_id,
            date.today().isoformat(),
            config["model"],
            config["dataset"],
            config["seed"],
            config["epochs"],
            config["batch_size"],
            config["lr"],
            config["base_filters"],
            f"{val_iou:.4f}"
        ])
    print(f"Logged to {log_path}")


# ── Training function ────────────────────────────────────────────────────────

def train(config):
    """
    Full training loop for one experiment run.
    config: dictionary containing all hyperparameters
    """

    # ── Reproducibility ──
    # Setting the seed ensures the same results every time you run with same seed
    torch.manual_seed(config["seed"])

    # ── Device ──
    # Use Apple GPU (MPS) if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ──
    train_loader, val_loader = get_dataloaders(
        data_dir   = config["data_dir"],
        image_size = config["image_size"],
        batch_size = config["batch_size"],
        seed       = config["seed"],
    )

    # ── Model ──
    model = MicroUNet(
        in_channels  = 3,
        out_channels = 1,
        base_filters = config["base_filters"],
    ).to(device)  # .to(device) moves model to GPU

    print(f"Model parameters: {count_parameters(model):,}")

    # ── Loss function ──
    # BCELoss = Binary Cross Entropy Loss
    # Measures how different the predicted mask is from the true mask
    # Perfect prediction = loss of 0, terrible prediction = high loss
    criterion = nn.BCELoss()

    # ── Optimizer ──
    # Adam adjusts learning rate automatically — standard choice
    # lr (learning rate) controls how big each update step is
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ── Training loop ──
    best_val_iou = 0.0

    for epoch in range(1, config["epochs"] + 1):

        # --- Training phase ---
        model.train()  # puts model in training mode (enables BatchNorm, Dropout etc.)
        train_loss = 0.0

        for images, masks in train_loader:
            # Move data to same device as model
            images = images.to(device)
            masks  = masks.to(device)

            # Zero gradients from previous step
            # (PyTorch accumulates gradients by default, we reset each step)
            optimizer.zero_grad()

            # Forward pass: feed images through model to get predictions
            predictions = model(images)

            # Calculate loss: how wrong are our predictions?
            loss = criterion(predictions, masks)

            # Backward pass: calculate gradients (how to adjust each parameter)
            loss.backward()

            # Update parameters using gradients
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()   # puts model in evaluation mode (disables dropout etc.)
        val_iou = 0.0

        with torch.no_grad():  # don't calculate gradients during validation
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                predictions = model(images)
                val_iou += compute_iou(predictions, masks)

        avg_val_iou = val_iou / len(val_loader)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou

        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f} | "
              f"Best IoU: {best_val_iou:.4f}")

    return best_val_iou


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Experiment 001-003: Baseline MicroUNet, 3 seeds for statistical reliability
    # Hypothesis: A 4-layer 8-filter SeparableConv U-Net with BatchNorm
    # achieves >0.7 IoU on BAGLS when trained for 10 epochs.
    base_config = {
        "model"       : "MicroUNet",
        "dataset"     : "BAGLS-test",
        "data_dir"    : "data/test",
        "image_size"  : 256,
        "base_filters": 8,
        "epochs"      : 10,
        "batch_size"  : 8,
        "lr"          : 1e-3,
    }

    seeds   = [42, 43, 44]
    run_ids = ["001", "002", "003"]

    results = []

    for seed, run_id in zip(seeds, run_ids):
        config = {**base_config, "seed": seed}

        print("=" * 50)
        print(f"Run ID: {run_id} | Seed: {seed}")
        print("=" * 50)

        save_config(config, run_id)
        best_val_iou = train(config)
        log_experiment(run_id, config, best_val_iou)
        results.append(best_val_iou)

        print(f"Run {run_id} complete. Best Val IoU: {best_val_iou:.4f}\n")

    # Summary across all seeds
    import statistics
    mean_iou = statistics.mean(results)
    std_iou  = statistics.stdev(results)

    print("=" * 50)
    print(f"FINAL SUMMARY (3 seeds)")
    print(f"IoU per seed: {[f'{r:.4f}' for r in results]}")
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print("=" * 50)