from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

import dyplom_mag.target_augment.net as net
from dyplom_mag.target_augment.function import adaptive_instance_normalization, coral

BASE_DIR = os.path.dirname(__file__)


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer."""
    # Basic options
    content: Optional[str] = None
    content_dir: Optional[str] = None
    style: Optional[str] = None
    style_dir: Optional[str] = None
    vgg: str = os.path.join(BASE_DIR, "pre_trained/vgg16_ori.pth")
    decoder: str = ''
    fc1: str = ''
    fc2: str = ''
    
    # Additional options
    save_ext: str = '.jpg'
    output: str = 'output'
    
    # Advanced options
    preserve_color: bool = False
    alpha: float = 1.0
    style_interpolation_weights: str = ''
    device: Union[str, int] = 0
    
    # Transform options
    content_size: int = 0
    style_size: int = 0
    crop: bool = False


def test_transform(size, crop):
    """Create a transform pipeline for test images."""
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, fc1, fc2, alpha=1.0,
                   interpolation_weights=None, device=None):
    """Perform style transfer between content and style images."""
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f, fc1, fc2)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f, fc1, fc2)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def preprocess(transform, path):
    """Preprocess an image for style transfer."""
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    img = Image.open(str(path)).convert('RGB')
    img = transform(img)
    img = np.array(img)
    img = img[:, :, ::-1]
    img = img.astype(np.float32, copy=False)
    img -= pixel_means
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img, pixel_means


def save_output_image(output, output_name, pixel_means):
    """Save the output image."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    # Save the image
    Image.fromarray((output[0].permute(1, 2, 0).numpy() + pixel_means)[:, :, ::-1].clip(0, 255).astype(np.uint8)).save(output_name)


def load_models(config, device):
    """Load the neural network models."""
    decoder = net.decoder
    vgg = net.vgg
    fc1 = net.fc1
    fc2 = net.fc2

    decoder.eval()
    vgg.eval()
    fc1.eval()
    fc2.eval()

    decoder.load_state_dict(torch.load(config.decoder))
    fc1.load_state_dict(torch.load(config.fc1))
    fc2.load_state_dict(torch.load(config.fc2))

    vgg.load_state_dict(torch.load(config.vgg)['model'])
    vgg = nn.Sequential(*list(vgg.children())[:19])

    vgg.to(device)
    decoder.to(device)
    fc1.to(device)
    fc2.to(device)
    
    return vgg, decoder, fc1, fc2


def process_style_transfer(config: StyleTransferConfig):
    """Process style transfer based on configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.device).isdigit() else "cpu")
    if str(config.device).isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    
    # Make sure output directory exists
    os.makedirs(config.output, exist_ok=True)
    
    # Validate inputs
    if not (config.content or config.content_dir):
        raise ValueError("Either --content or --content_dir should be given.")
        
    if not (config.style or config.style_dir):
        raise ValueError("Either --style or --style_dir should be given.")
    
    # Get content paths
    if config.content:
        content_paths = [Path(config.content)]
    else:
        content_dir = Path(config.content_dir)
        content_paths = [f for f in content_dir.glob('*')]
    
    # Get style paths and check if we need to do interpolation
    do_interpolation = False
    if config.style:
        style_paths = config.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(config.style)]
        else:
            do_interpolation = True
            if not config.style_interpolation_weights:
                raise ValueError('Please specify interpolation weights')
                
            weights = [int(i) for i in config.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(config.style_dir)
        style_paths = [f for f in style_dir.glob('*')]
    
    # Load models
    vgg, decoder, fc1, fc2 = load_models(config, device)
    
    # Create transforms
    content_tf = test_transform(config.content_size, config.crop)
    style_tf = test_transform(config.style_size, config.crop)
    
    # Process each content image
    for content_path in content_paths:
        if do_interpolation:  # one content image, N style image
            style_images = []
            for p in style_paths:
                style_img, _ = preprocess(style_tf, p)
                style_images.append(style_img)
                
            style = torch.stack(style_images)
            content, pixel_means = preprocess(content_tf, content_path)
            content = content.unsqueeze(0).expand_as(style)
            
            style = style.to(device)
            content = content.to(device)
            
            with torch.no_grad():
                output = style_transfer(
                    vgg, decoder, content, style, fc1, fc2,
                    config.alpha, interpolation_weights, device
                )
                
            output = output.cpu()
            output_name = os.path.join(config.output, f'{content_path.stem}_interpolation{config.save_ext}')
            save_output_image(output, output_name, pixel_means)
        else:  # process one content and one style
            for style_path in style_paths:
                content, pixel_means = preprocess(content_tf, content_path)
                style, _ = preprocess(style_tf, style_path)
                
                if config.preserve_color:
                    style = coral(style, content)
                    
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                
                with torch.no_grad():
                    output = style_transfer(
                        vgg, decoder, content, style, fc1, fc2,
                        config.alpha, device=device
                    )
                    
                output = output.cpu()
                output_name = os.path.join(
                    config.output, 
                    f'{content_path.stem}{config.save_ext}'
                )
                save_output_image(output, output_name, pixel_means)


def main(config: StyleTransferConfig):
    """Main function to run style transfer."""
    process_style_transfer(config)


# Allow script to be run standalone for backward compatibility
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, help='File path to the content image')
    parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str, help='File path to the style image, or multiple style images separated by commas if you want to do style interpolation or spatial control')
    parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='')
    parser.add_argument('--decoder', type=str, default='')
    parser.add_argument('--fc1', type=str, default='')
    parser.add_argument('--fc2', type=str, default='')
    
    # Additional options
    parser.add_argument('--content_size', type=int, default=0, help='Size of content images')
    parser.add_argument('--style_size', type=int, default=0, help='Size of style images')
    parser.add_argument('--crop', action='store_true', help='If specified, crop images to square')
    parser.add_argument('--save_ext', default='.jpg', help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')
    
    # Advanced options
    parser.add_argument('--preserve_color', action='store_true', help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0, help='The weight that controls the degree of stylization. Should be between 0 and 1')
    parser.add_argument('--style_interpolation_weights', type=str, default='', help='The weight for blending the style of multiple style images')
    parser.add_argument('--device', type=str, default=0)
    
    args = parser.parse_args()
    
    # Convert args to config
    config = StyleTransferConfig(
        content=args.content,
        content_dir=args.content_dir,
        style=args.style,
        style_dir=args.style_dir,
        vgg=args.vgg,
        decoder=args.decoder,
        fc1=args.fc1,
        fc2=args.fc2,
        content_size=args.content_size,
        style_size=args.style_size,
        crop=args.crop,
        save_ext=args.save_ext,
        output=args.output,
        preserve_color=args.preserve_color,
        alpha=args.alpha,
        style_interpolation_weights=args.style_interpolation_weights,
        device=args.device
    )
    
    main(config)