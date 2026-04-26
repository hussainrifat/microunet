# visualize.py
# Generates two visualizations:
# 1. Sample predictions: image | ground truth mask | predicted mask
# 2. Training loss curve across epochs
# Run: python3 visualize.py

import os
import torch
import matplotlib
matplotlib.use('Agg')  # no display needed, saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.dataset import get_dataloaders
from models.unet import MicroUNet


def visualize_predictions(model, val_loader, device, num_samples=4, save_path="results/predictions.png"):
    """
    Shows num_samples examples of: input image | ground truth | prediction
    This is the most important visualization — shows the model actually works.
    """
    model.eval()  # evaluation mode — no gradient tracking needed

    # Collect one batch of samples
    images, masks = next(iter(val_loader))
    images = images.to(device)
    masks  = masks.to(device)

    # Get model predictions
    with torch.no_grad():  # don't compute gradients — saves memory
        predictions = model(images)

    # Move everything back to CPU for plotting
    images      = images.cpu()
    masks       = masks.cpu()
    predictions = predictions.cpu()

    # Create a figure with num_samples rows, 3 columns
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    fig.suptitle("MicroUNet Predictions on BAGLS\n(4 layers, 8 filters, 63,628 params)", 
                 fontsize=13, fontweight='bold')

    for i in range(num_samples):
        # Column 0: input image
        # images[i] has shape [3, 256, 256] — we need [256, 256, 3] for matplotlib
        img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image" if i == 0 else "")
        axes[i, 0].axis('off')  # hide axis ticks

        # Column 1: ground truth mask
        # masks[i] has shape [1, 256, 256] — squeeze removes the channel dim
        gt = masks[i].squeeze().numpy()
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis('off')

        # Column 2: predicted mask (binarized at 0.5 threshold)
        pred = (predictions[i].squeeze() > 0.5).float().numpy()
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction" if i == 0 else "")
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved predictions to {save_path}")
    plt.close()


def plot_training_curve(train_losses, val_ious, save_path="results/training_curve.png"):
    """
    Plots training loss and validation IoU across epochs.
    Shows the model is learning — loss goes down, IoU goes up.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MicroUNet Training — BAGLS Dataset (Seed 42)", 
                 fontsize=13, fontweight='bold')

    # Left plot: training loss
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right plot: validation IoU
    ax2.plot(epochs, val_ious, 'g-o', linewidth=2, markersize=4, label='Val IoU')
    ax2.axhline(y=max(val_ious), color='r', linestyle='--', 
                alpha=0.5, label=f'Best IoU: {max(val_ious):.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curve to {save_path}")
    plt.close()


def train_and_visualize():
    """
    Trains the model for 10 epochs while recording metrics,
    then generates both visualizations.
    """
    import torch.nn as nn

    # Setup
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        data_dir   = "data/test",
        image_size = 256,
        batch_size = 8,
        seed       = 42,
    )

    # Model
    model = MicroUNet(in_channels=3, out_channels=1, base_filters=8).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Track metrics for plotting
    train_losses = []
    val_ious     = []

    print("Training for 10 epochs...")

    for epoch in range(1, 11):

        # Training phase
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation phase
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                # compute IoU
                preds_bin    = (predictions > 0.5).float()
                intersection = (preds_bin * masks).sum(dim=(1,2,3))
                union        = (preds_bin + masks).clamp(0,1).sum(dim=(1,2,3))
                iou = torch.where(union==0, torch.ones_like(intersection), 
                                  intersection/union)
                val_iou += iou.mean().item()

        avg_iou = val_iou / len(val_loader)
        val_ious.append(avg_iou)

        print(f"Epoch {epoch:02d}/10 | Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curve(train_losses, val_ious)
    visualize_predictions(model, val_loader, device)
    print("\nDone! Check the results/ folder.")


if __name__ == "__main__":
    train_and_visualize()