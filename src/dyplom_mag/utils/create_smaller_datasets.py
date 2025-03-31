import math
import os
import shutil


def create_multi_image_datasets(
    dataset_folder, output_base_path, batch_size=10, image_split="train"
):
    """
    Splits a dataset into smaller datasets, each containing a specified number of images.

    Args:
        dataset_folder (str): Path to the original dataset folder.
        output_base_path (str): Base path where mini-datasets will be created.
        batch_size (int): Number of images to include in each mini-dataset.
        image_split (str): Which split to use from the original dataset (train, valid, or test).

    Returns:
        int: Number of mini-datasets created.
    """
    # Build source paths from the dataset folder
    original_config_path = os.path.join(dataset_folder, "data.yaml")
    original_images_folder = os.path.join(dataset_folder, image_split, "images")
    original_labels_folder = os.path.join(dataset_folder, image_split, "labels")

    # Check that the necessary files/folders exist
    if not os.path.exists(original_config_path):
        raise ValueError(f"data.yaml not found at {original_config_path}")
    if not os.path.exists(original_images_folder):
        raise ValueError(f"Images folder not found at {original_images_folder}")

    # List all image files in the source images folder
    image_files = [
        f
        for f in os.listdir(original_images_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    image_files.sort()  # Process files in a sorted order

    # Calculate how many mini-datasets we'll create
    num_datasets = math.ceil(len(image_files) / batch_size)

    # Ensure the output base path exists
    os.makedirs(output_base_path, exist_ok=True)

    # Create each mini-dataset
    for dataset_idx in range(num_datasets):
        # Calculate start and end indices for this batch
        start_idx = dataset_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        current_batch = image_files[start_idx:end_idx]

        # Create a mini-dataset folder
        mini_dataset_folder = os.path.join(
            output_base_path, f"dataset_{dataset_idx + 1:04d}"
        )

        # Create required subdirectories for each split
        for split in ["train", "valid", "test"]:
            os.makedirs(
                os.path.join(mini_dataset_folder, split, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(mini_dataset_folder, split, "labels"), exist_ok=True
            )

        # Process each image in this batch
        for image_file in current_batch:
            # Copy the image to all splits
            src_image_path = os.path.join(original_images_folder, image_file)
            for split in ["train", "valid", "test"]:
                dst_image_path = os.path.join(
                    mini_dataset_folder, split, "images", image_file
                )
                shutil.copy(src_image_path, dst_image_path)

            # Process the corresponding label file
            base_name, _ = os.path.splitext(image_file)
            label_file = base_name + ".txt"
            src_label_path = os.path.join(original_labels_folder, label_file)

            for split in ["train", "valid", "test"]:
                dst_label_path = os.path.join(
                    mini_dataset_folder, split, "labels", label_file
                )
                if os.path.exists(src_label_path):
                    shutil.copy(src_label_path, dst_label_path)
                else:
                    # If the label does not exist, create an empty file
                    open(dst_label_path, "a").close()

        # Copy the original data.yaml file to the root of the mini-dataset
        dst_config_path = os.path.join(mini_dataset_folder, "data.yaml")
        shutil.copy(original_config_path, dst_config_path)

        print(
            f"Created mini-dataset {dataset_idx + 1}/{num_datasets}: {mini_dataset_folder} "
            f"with {len(current_batch)} images"
        )

    return num_datasets
