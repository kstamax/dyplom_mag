import argparse
import os
import random

import numpy as np
from PIL import Image


def get_random_images(folder_path, num_images=5, ext=".jpg"):
    """
    Get random image paths from a folder with the specified extension.

    Args:
        folder_path (str): Path to the folder containing images
        num_images (int): Number of random images to select
        ext (str): File extension to filter images

    Returns:
        list: List of paths to randomly selected images
    """
    # Get all files with the specified extension
    all_images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(ext)
    ]

    if not all_images:
        raise ValueError(f"No images with extension {ext} found in {folder_path}")

    if len(all_images) < num_images:
        print(f"Warning: Found only {len(all_images)} images, using all of them")
        return all_images

    # Select random images
    return random.sample(all_images, num_images)


def compute_mean_style_image(image_paths, output_size=(512, 512)):
    """
    Compute the mean style image from a list of image paths.

    Args:
        image_paths (list): List of paths to style images
        output_size (tuple): Size (width, height) of the output image

    Returns:
        PIL.Image: Mean style image
    """
    # Load and resize all images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(output_size, Image.LANCZOS)
            images.append(np.array(img))
        except Exception as e:
            print(f"Error processing image {path}: {e}")

    if not images:
        raise ValueError("No valid images found")

    # Compute the mean image
    images_array = np.stack(images, axis=0)
    mean_image_array = np.mean(images_array, axis=0).astype(np.uint8)

    # Convert back to PIL Image
    mean_image = Image.fromarray(mean_image_array)
    return mean_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate mean style image for AdaIN model"
    )
    parser.add_argument(
        "folder_path", type=str, help="Path to folder containing style images"
    )
    parser.add_argument(
        "--output", type=str, default="mean_style.jpg", help="Path to save output image"
    )
    parser.add_argument(
        "--num_images", type=int, default=5, help="Number of random images to select"
    )
    parser.add_argument("--width", type=int, default=800, help="Output image width")
    parser.add_argument("--height", type=int, default=600, help="Output image height")

    args = parser.parse_args()

    # Get random image paths
    image_paths = get_random_images(args.folder_path, args.num_images)
    print(f"Selected {len(image_paths)} random images:")
    for path in image_paths:
        print(f"  - {path}")

    # Compute mean style image
    mean_image = compute_mean_style_image(image_paths, (args.width, args.height))

    # Save the mean style image
    mean_image.save(args.output)
    print(f"Mean style image saved to: {args.output}")

    return mean_image


if __name__ == "__main__":
    main()
