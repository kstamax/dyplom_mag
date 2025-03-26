import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

import dyplom_mag.target_augment.net as net
from dyplom_mag.target_augment.sampler import InfiniteSamplerWrapper

# Disable DecompressionBombError and truncated image warnings
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(__file__)


@dataclass
class TrainingConfig:
    """Configuration class for style transfer training"""

    # Directories
    content_dir: str
    style_dir: str = os.path.join(BASE_DIR, "dataset/style")
    vgg_path: str = os.path.join(BASE_DIR, "pre_trained/vgg16_ori.pth")
    save_dir: str = os.path.join(BASE_DIR, "models")

    # Training hyperparameters
    max_iter: int = 160000
    batch_size: int = 8
    n_threads: int = 0
    save_model_interval: int = -1
    lr_decay: float = 5e-5

    # Learning rates
    lr_decoder: float = 1e-4
    lr_fcs: float = 1e-4

    # Loss weights
    style_weight: float = 50
    content_weight: float = 1
    content_style_weight: float = 1
    constrain_weight: float = 1
    before_fcs_steps: int = 0

    # Device configuration
    device: str = "cuda"


def train_transform():
    """Create image transformation for training"""
    transform_list = [
        transforms.Resize(size=(600, 800)),
        transforms.RandomCrop(128),
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        # Add error handling for path discovery
        try:
            self.paths = list(Path(self.root).glob("*"))
            if not self.paths:
                raise ValueError(f"No image files found in directory: {root}")
        except Exception as e:
            print(f"Error discovering image files: {e}")
            self.paths = []

        self.transform = transform
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    def __getitem__(self, index):
        # Robust image loading
        try:
            path = self.paths[index % len(self.paths)]  # Prevent index out of bounds
            img = Image.open(str(path)).convert("RGB")
            img = self.transform(img)
            img = np.array(img)
            img = img[:, :, ::-1]
            if np.random.rand() >= 0.5:
                img = img[:, ::-1, :]
            img = img.astype(np.float32, copy=False)
            img -= self.pixel_means
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy tensor to prevent complete failure
            return torch.zeros((3, 128, 128), dtype=torch.float32)

    def __len__(self):
        return len(self.paths)


def get_device(device_arg):
    """
    Parse device argument and return appropriate torch device
    Args:
        device_arg: Either 'cuda', 'cpu', 'mps', or specific device like 'cuda:0'
    Returns:
        torch.device object
    """
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    elif device_arg == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return torch.device("cpu")
    elif device_arg.startswith("cuda:") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    else:
        return torch.device(device_arg)


def adjust_learning_rate(init_lr, optimizer, iteration_count):
    """Adjust learning rate during training"""
    lr = init_lr / (1.0 + iteration_count * 5e-5)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_style_transfer(config: TrainingConfig):
    """
    Main training pipeline for style transfer

    Args:
        config (TrainingConfig): Configuration for training
    """
    # Enable cudnn benchmark if CUDA is available
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # Determine device
    if config.device.startswith("cuda:"):
        device_idx = config.device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
        device = get_device("cuda")
    else:
        device = get_device(config.device)

    print(f"Using device: {device}")

    # Adjust number of workers based on device
    if device.type == "cpu":
        import multiprocessing

        config.n_threads = min(config.n_threads, multiprocessing.cpu_count())
    elif device.type == "mps":
        # MPS often works better with fewer threads
        config.n_threads = min(config.n_threads, 4)

    # Create save directory if it doesn't exist
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize network components
    decoder = net.decoder
    vgg = net.vgg
    fc1 = net.fc1
    fc2 = net.fc2

    # Load VGG with appropriate map_location
    vgg.load_state_dict(torch.load(config.vgg_path, map_location=device)["model"])
    vgg = nn.Sequential(*list(vgg.children())[:19])

    # Create network
    network = net.Net(vgg, decoder, fc1, fc2)
    network.train()
    network.to(device)

    # Create data transformations
    content_tf = train_transform()
    style_tf = train_transform()

    # Create datasets
    content_dataset = FlatFolderDataset(config.content_dir, content_tf)
    style_dataset = FlatFolderDataset(config.style_dir, style_tf)

    # Create data loaders
    content_loader = data.DataLoader(
        content_dataset,
        batch_size=config.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=config.n_threads,
        pin_memory=device.type == "cuda",
        persistent_workers=config.n_threads > 0,
    )
    style_loader = data.DataLoader(
        style_dataset,
        batch_size=config.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=config.n_threads,
        pin_memory=device.type == "cuda",
        persistent_workers=config.n_threads > 0,
    )

    # Create iterators with error handling
    try:
        content_iter = iter(content_loader)
        style_iter = iter(style_loader)
    except Exception as e:
        print(f"Error creating data iterators: {e}")
        raise

    # Create optimizers
    optimizer1 = torch.optim.Adam(
        itertools.chain(
            *[
                network.dec_1.parameters(),
                network.dec_2.parameters(),
                network.dec_3.parameters(),
                network.dec_4.parameters(),
            ]
        ),
        lr=config.lr_decoder,
    )
    optimizer2 = torch.optim.Adam(
        itertools.chain(*[network.fc1.parameters(), network.fc2.parameters()]),
        lr=config.lr_fcs,
    )

    # Main training loop with robust error handling
    try:
        for i in tqdm(range(config.max_iter)):
            try:
                # Adjust decoder learning rate
                adjust_learning_rate(config.lr_decoder, optimizer1, iteration_count=i)

                # Load data
                try:
                    content_images = next(content_iter).to(device)
                    style_images = next(style_iter).to(device)
                except StopIteration:
                    # Recreate iterators if they are exhausted
                    content_iter = iter(content_loader)
                    style_iter = iter(style_loader)
                    content_images = next(content_iter).to(device)
                    style_images = next(style_iter).to(device)

                # First stage of training
                loss_c, loss_const = network(content_images, style_images, flag=0)
                loss_c = config.content_weight * loss_c
                loss_const = config.constrain_weight * loss_const
                loss = loss_c + loss_const

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()

                # Second stage of training
                if i >= config.before_fcs_steps:
                    adjust_learning_rate(
                        config.lr_fcs,
                        optimizer2,
                        iteration_count=i - config.before_fcs_steps,
                    )
                    loss_s_1, loss_s_2 = network(content_images, style_images, flag=1)
                    loss_s_1 = config.style_weight * loss_s_1
                    loss_s_2 = config.content_style_weight * loss_s_2
                    loss = loss_s_1 + loss_s_2

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()

                # Save model checkpoints
                save = False
                if config.save_model_interval != -1:
                    if (i + 1) % config.save_model_interval == 0 or (
                        i + 1
                    ) == config.max_iter:
                        save = True
                else:
                    if (i + 1) == config.max_iter:
                        save = True

                if save:
                    # Move models to CPU before saving to ensure compatibility across devices
                    state_dict = net.decoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device("cpu"))
                    torch.save(
                        state_dict,
                        os.path.join(config.save_dir, f"decoder_iter_{i + 1}.pth"),
                    )

                    state_dict = net.fc1.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device("cpu"))
                    torch.save(
                        state_dict,
                        os.path.join(config.save_dir, f"fc1_iter_{i + 1}.pth"),
                    )

                    state_dict = net.fc2.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device("cpu"))
                    torch.save(
                        state_dict,
                        os.path.join(config.save_dir, f"fc2_iter_{i + 1}.pth"),
                    )

            except Exception as iter_error:
                print(f"Error in training iteration {i}: {iter_error}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")
        # Save the current state on keyboard interrupt
        for model_name, model in [
            ("decoder", net.decoder),
            ("fc1", net.fc1),
            ("fc2", net.fc2),
        ]:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device("cpu"))
            torch.save(
                state_dict,
                os.path.join(config.save_dir, f"{model_name}_interrupted.pth"),
            )
        print("Interrupted model saved.")
