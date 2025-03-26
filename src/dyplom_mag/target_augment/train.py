import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net
from sampler import InfiniteSamplerWrapper
import numpy as np
import itertools
import sys

# Improved error handling and logging
def add_error_handling():
    def exception_hook(exctype, value, traceback):
        print(f"Uncaught exception: {exctype.__name__}: {value}", file=sys.stderr)
        sys.__excepthook__(exctype, value, traceback)
    sys.excepthook = exception_hook

# Only enable cudnn benchmark if CUDA is available
if torch.cuda.is_available():
    cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():
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
            self.paths = list(Path(self.root).glob('*'))
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
            img = Image.open(str(path)).convert('RGB')
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

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(init_lr, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = init_lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_device(device_arg):
    """
    Parse device argument and return appropriate torch device
    Args:
        device_arg: Either 'cuda', 'cpu', 'mps', or specific device like 'cuda:0'
    Returns:
        torch.device object
    """
    if device_arg == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')
    elif device_arg == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return torch.device('cpu')
    elif device_arg.startswith('cuda:') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')
    else:
        return torch.device(device_arg)

def main():
    # Add error handling
    add_error_handling()

    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./dataset/style',
                        help='Directory path to a batch of style images')

    parser.add_argument('--vgg', type=str, default='./pre_trained/vgg16_ori.pth')

    parser.add_argument('--save_dir', default='./models',
                        help='Directory to save the model')
    parser.add_argument('--n_threads', type=int, default=0)  # Change to 0 for debugging
    parser.add_argument('--save_model_interval', type=int, default=-1)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--lr_decoder', type=float, default=1e-4)
    parser.add_argument('--lr_fcs', type=float, default=1e-4)
    parser.add_argument('--style_weight', type=float, default=50)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--content_style_weight', type=float, default=1)
    parser.add_argument('--constrain_weight', type=float, default=1)
    parser.add_argument('--before_fcs_steps', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training: cuda, cpu, or mps')
    parser.add_argument('--adjust_workers', action='store_true',
                        help='Automatically adjust number of workers based on device')
    global args
    args = parser.parse_args()

    # Get device
    if args.device.startswith('cuda:'):
        device_idx = args.device.split(':')[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
        device = get_device('cuda')
    else:
        device = get_device(args.device)

    print(f"Using device: {device}")

    # Adjust number of workers based on device if requested
    if args.adjust_workers:
        if device.type == 'cpu':
            import multiprocessing
            args.n_threads = min(args.n_threads, multiprocessing.cpu_count())
        elif device.type == 'mps':
            # MPS often works better with fewer threads
            args.n_threads = min(args.n_threads, 4)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    decoder = net.decoder
    vgg = net.vgg
    fc1 = net.fc1
    fc2 = net.fc2

    # Load model with appropriate map_location
    vgg.load_state_dict(torch.load(args.vgg, map_location=device)['model'])
    vgg = nn.Sequential(*list(vgg.children())[:19])

    network = net.Net(vgg, decoder, fc1, fc2)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    # Adjust data loading for MPS and CPU
    content_loader = data.DataLoader(
        content_dataset, 
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.n_threads > 0
    )
    style_loader = data.DataLoader(
        style_dataset, 
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.n_threads > 0
    )

    # Create iterators with error handling
    try:
        content_iter = iter(content_loader)
        style_iter = iter(style_loader)
    except Exception as e:
        print(f"Error creating data iterators: {e}")
        raise

    optimizer1 = torch.optim.Adam(
        itertools.chain(*[network.dec_1.parameters(), network.dec_2.parameters(), 
                         network.dec_3.parameters(), network.dec_4.parameters()]), 
        lr=args.lr_decoder)
    optimizer2 = torch.optim.Adam(
        itertools.chain(*[network.fc1.parameters(), network.fc2.parameters()]), 
        lr=args.lr_fcs)

    # Main training loop with more robust error handling
    try:
        for i in tqdm(range(args.max_iter)):
            try:
                adjust_learning_rate(args.lr_decoder, optimizer1, iteration_count=i)

                # Use a try-except block for data loading
                try:
                    content_images = next(content_iter).to(device)
                    style_images = next(style_iter).to(device)
                except StopIteration:
                    # Recreate iterators if they are exhausted
                    content_iter = iter(content_loader)
                    style_iter = iter(style_loader)
                    content_images = next(content_iter).to(device)
                    style_images = next(style_iter).to(device)

                loss_c, loss_const = network(content_images, style_images, flag=0)
                loss_c = args.content_weight * loss_c
                loss_const = args.constrain_weight * loss_const
                loss = loss_c + loss_const

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()

                if i >= args.before_fcs_steps:
                    adjust_learning_rate(args.lr_fcs, optimizer2, iteration_count=i-args.before_fcs_steps)
                    loss_s_1, loss_s_2 = network(content_images, style_images, flag=1)
                    loss_s_1 = args.style_weight * loss_s_1
                    loss_s_2 = args.content_style_weight * loss_s_2
                    loss = loss_s_1 + loss_s_2

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()

                # Save model checkpoints
                save = False
                if args.save_model_interval != -1:
                    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                        save = True
                else:
                    if (i + 1) == args.max_iter:
                        save = True
                        
                if save:
                    # Move models to CPU before saving to ensure compatibility across devices
                    state_dict = net.decoder.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, os.path.join(args.save_dir, f'decoder_iter_{i + 1}.pth'))
                    
                    state_dict = net.fc1.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, os.path.join(args.save_dir, f'fc1_iter_{i + 1}.pth'))
                    
                    state_dict = net.fc2.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].to(torch.device('cpu'))
                    torch.save(state_dict, os.path.join(args.save_dir, f'fc2_iter_{i + 1}.pth'))
                    
            except Exception as iter_error:
                print(f"Error in training iteration {i}: {iter_error}")
                # Optional: break or continue based on the type of error
                
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")
        # Save the current state on keyboard interrupt
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'decoder_interrupted.pth'))
        
        state_dict = net.fc1.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'fc1_interrupted.pth'))
        
        state_dict = net.fc2.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, os.path.join(args.save_dir, 'fc2_interrupted.pth'))
        print("Interrupted model saved.")

if __name__ == '__main__':
    main()