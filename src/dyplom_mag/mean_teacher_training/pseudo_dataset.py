import os
import shutil
from PIL import Image
import copy
from dyplom_mag.target_augment.test import process_style_transfer, StyleTransferConfig
from dyplom_mag.utils.plot_predictions import plot_predictions_and_ground_truth

BASE_DIR = os.path.dirname(__file__)


def save_pseudo_labels(result, image_file, labels_dir, conf_threshold):
    """
    Save pseudo-labels in YOLO format for one image.

    Args:
        result: A Results object with attribute boxes.data: tensor [N,6] (x1, y1, x2, y2, conf, cls)
        image_file: File path of the image.
        labels_dir: Directory to save the .txt file.
        conf_threshold: Confidence threshold to filter detections.
    """
    base = os.path.splitext(os.path.basename(image_file))[0]
    label_path = os.path.join(labels_dir, base + ".txt")
    # If there are no detections, save an empty file.
    if result.boxes is None or result.boxes.data.numel() == 0:
        open(label_path, 'w').close()
        return
    boxes = result.boxes.data.cpu()  # tensor of shape [N,6]
    # Open the image to get its actual dimensions.
    with Image.open(image_file) as img:
        width, height = img.size
    with open(label_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf < conf_threshold:
                continue
            # Convert box to YOLO format (normalized center_x, center_y, width, height)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

def prepare_pseudo_dataset(
    data_dir,
    teacher_model,
    pseudo_dataset_dir="",
    style_dir="/content/style",
    im_type='valid',
    conf_threshold=0.25,
    iou=0.3
):
    if not pseudo_dataset_dir:
        pseudo_dataset_dir = os.path.join(BASE_DIR, "pseudo_dataset")
    # Load original data.yaml
    data_yaml_path = os.path.join(data_dir, "data.yaml")

    # Assume images are in <data_dir>/valid/images
    original_train_dir = os.path.join(data_dir, im_type, "images")
    print("Original training images at:", original_train_dir)

    # List image files in original training folder
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [os.path.join(original_train_dir, f) for f in os.listdir(original_train_dir)
                   if f.lower().endswith(img_extensions)]
    print(f"Found {len(image_files)} images.")

    # Instead of using teacher_model directly, create a deepcopy for inference.
    teacher_inference = copy.deepcopy(teacher_model)
    teacher_inference.model.eval()  # set to eval mode for predictions

    # Run teacher predictions on all images.
    print("Running teacher predictions...")
    results = teacher_inference.predict(source=image_files, conf=conf_threshold, iou=iou)

    # Create new pseudo-labeled dataset folder structure.
    if os.path.exists(pseudo_dataset_dir):
        shutil.rmtree(pseudo_dataset_dir)
    new_images_dir = os.path.join(pseudo_dataset_dir, im_type, "images")
    new_labels_dir = os.path.join(pseudo_dataset_dir, im_type, "labels")
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)

    # Save pseudo-labels for each image.
    for img_file, res in zip(image_files, results):
        save_pseudo_labels(res, img_file, new_labels_dir, conf_threshold)

    shutil.copy(data_yaml_path, f"{pseudo_dataset_dir}/data.yaml")

    config = StyleTransferConfig(
        content_dir=original_train_dir,
        style_dir=style_dir,
        decoder=os.path.join(BASE_DIR, "..", "target_augment", "models", "decoder.pth"),
        fc1=os.path.join(BASE_DIR, "..", "target_augment", "models", "fc1.pth"),
        fc2=os.path.join(BASE_DIR, "..", "target_augment", "models", "fc2.pth"),
        output=new_images_dir,
        alpha=0.2,
    )
    process_style_transfer(config)
    # Optional: Plot predictions on a sample image for inspection.
    sample_img_path = os.path.join(new_images_dir, os.path.basename(image_files[5]))
    sample_label_path = os.path.join(new_labels_dir, os.path.splitext(os.path.basename(image_files[5]))[0] + ".txt")
    plot_predictions_and_ground_truth(teacher_inference, sample_img_path, sample_label_path, device="cuda")
