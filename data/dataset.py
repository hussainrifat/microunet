# dataset.py
# This file teaches PyTorch how to load BAGLS images and masks

import os                          # for navigating file paths
from PIL import Image              # for opening image files
import torch                       # the deep learning framework
from torch.utils.data import Dataset, DataLoader  # base classes for data loading
import torchvision.transforms as transforms        # for resizing/converting images


class BAGLSDataset(Dataset):
    """
    A Dataset class for the BAGLS laryngeal endoscopy dataset.
    Every PyTorch dataset must implement three methods: __init__, __len__, __getitem__
    """

    def __init__(self, data_dir, image_size=256):
        """
        Called once when we create the dataset object.
        data_dir: path to the folder containing images and masks
        image_size: we resize all images to this size (256x256)
        """
        self.data_dir = data_dir
        self.image_size = image_size

        # Find all image files — these are .png files WITHOUT "_seg" in the name
        # Example: 0.png is an image, 0_seg.png is its mask
        all_files = os.listdir(data_dir)
        self.image_ids = [
            f.replace(".png", "")           # remove .png to get just the number e.g. "42"
            for f in all_files
            if f.endswith(".png") and "_seg" not in f  # only image files, not masks
        ]
        self.image_ids.sort()  # sort so order is consistent across runs

        # Define how to transform images: resize then convert to tensor
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # resize to 256x256
            transforms.ToTensor(),            # convert PIL image to PyTorch tensor
            # ToTensor also rescales pixel values from [0,255] to [0.0,1.0]
        ])

        # Define how to transform masks: resize (nearest neighbor to keep binary values)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        print(f"Dataset loaded: {len(self.image_ids)} images from {data_dir}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns one sample (image + mask) at position idx.
        PyTorch calls this automatically during training.
        """
        image_id = self.image_ids[idx]

        # Build file paths for image and mask
        image_path = os.path.join(self.data_dir, f"{image_id}.png")
        mask_path  = os.path.join(self.data_dir, f"{image_id}_seg.png")

        # Open image and convert to RGB (3 color channels)
        image = Image.open(image_path).convert("RGB")

        # Open mask and convert to L (grayscale, 1 channel)
        mask = Image.open(mask_path).convert("L")

        # Apply transforms
        image = self.image_transform(image)   # shape: [3, 256, 256]
        mask  = self.mask_transform(mask)     # shape: [1, 256, 256]

        # Binarize mask: any pixel > 0.5 becomes 1.0 (foreground), rest 0.0 (background)
        mask = (mask > 0.5).float()

        return image, mask


def get_dataloaders(data_dir, image_size=256, batch_size=8, val_split=0.2, seed=42):
    """
    Creates train and validation DataLoaders from a single folder.
    val_split=0.2 means 20% of data goes to validation, 80% to training.
    """
    # Create the full dataset
    full_dataset = BAGLSDataset(data_dir, image_size)

    # Calculate split sizes
    total = len(full_dataset)
    val_size   = int(total * val_split)   # 20% for validation
    train_size = total - val_size         # 80% for training

    # Split dataset randomly but reproducibly using the seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Create DataLoaders — these handle batching and shuffling automatically
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # shuffle training data every epoch
        num_workers=0,      # 0 = load data on main process (safe for Mac)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # no shuffle for validation — we want consistent results
        num_workers=0,
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")
    return train_loader, val_loader