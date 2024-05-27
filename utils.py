import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class SpriteDataset(Dataset):
    """Sprite dataset class for loading and transforming sprite images and labels."""

    def __init__(self, root, transform, target_transform):
        """
        Initialize the dataset.

        Parameters:
        - root: str, path to the directory containing the dataset files
        - transform: function, transformation to apply to the images
        - target_transform: function, transformation to apply to the labels
        """
        self.images = np.load(os.path.join(root, "Example_16x16.npy"))
        self.labels = np.load(os.path.join(root, "Example_labels_16x16.npy"))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
        - idx: int, index of the item to retrieve

        Returns:
        - image: transformed image at the given index
        - label: transformed label at the given index
        """
        image = self.transform(self.images[idx])
        label = self.target_transform(self.labels[idx])
        return image, label

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

def generate_animation(intermediate_samples, t_steps, fname, n_images_per_row=8):
    """
    Generates an animation and saves it as a gif file.

    Parameters:
    - intermediate_samples: list of tensor, intermediate samples to animate
    - t_steps: list of int, timesteps corresponding to the intermediate samples
    - fname: str, filename to save the animation as
    - n_images_per_row: int, number of images per row in the grid
    """
    # Create a grid of images for each set of intermediate samples
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        """Update the plot for each frame of the animation."""
        img_plot.set_array(intermediate_samples[frame])
        ax.set_title(f"T = {t_steps[frame]}")
        fig.tight_layout()
        return img_plot
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    # Save the animation as a gif
    ani.save(fname)

def get_custom_context(n_samples, n_classes, device):
    """
    Returns custom context in one-hot encoded form.

    Parameters:
    - n_samples: int, number of samples to generate
    - n_classes: int, number of classes for one-hot encoding
    - device: torch.device, device to create the tensor on

    Returns:
    - context: tensor, one-hot encoded context
    """
    context = []
    for i in range(n_classes - 1):
        context.extend([i] * (n_samples // n_classes))
    context.extend([n_classes - 1] * (n_samples - len(context)))
    
    # Convert to one-hot encoding and move to the specified device
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
