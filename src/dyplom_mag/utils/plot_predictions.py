import cv2
import matplotlib.pyplot as plt


def plot_predictions_and_ground_truth(model, image_path, label_path, device="cuda"):
    class_names = ["автомобіль", "вантажівка"]

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    height, width = original_image.shape[:2]

    image_pred = original_image.copy()
    image_gt = original_image.copy()

    results = model.predict(source=image_path, device=device)
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(
            image_pred, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
        )
        text = f"{label}: {conf:.2f}"
        cv2.putText(
            image_pred,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_str, x_center, y_center, box_w, box_h = parts[:5]
        try:
            class_id = int(cls_str)
        except ValueError:
            class_id = -1
        label = class_names[class_id] if 0 <= class_id < len(class_names) else cls_str

        x_center = float(x_center) * width
        y_center = float(y_center) * height
        box_w = float(box_w) * width
        box_h = float(box_h) * height

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(image_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_gt,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Predictions (Red Boxes)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth (Green Boxes)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def compare_model_predictions(model1, model2, image_path, device="cuda", 
                             left_title="Model 1 Predictions", right_title="Model 2 Predictions", 
                             main_title=None, class_names=["автомобіль", "вантажівка"]):
    """
    Compare predictions from two models on the same image.
    
    Args:
        model1: First model for prediction
        model2: Second model for prediction
        image_path (str): Path to the input image
        device (str): Device to run inference on (e.g., "cuda", "cpu")
        left_title (str): Title for the left plot (model1)
        right_title (str): Title for the right plot (model2)
        main_title (str): Optional main title for the entire figure
        class_names (list): List of class names for the models
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    
    image1 = original_image.copy()
    image2 = original_image.copy()
    
    # Get predictions from model 1
    results1 = model1.predict(source=image_path, device=device)
    for box in results1[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(
            image1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
        )
        text = f"{label}: {conf:.2f}"
        cv2.putText(
            image1,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    
    # Get predictions from model 2
    results2 = model2.predict(source=image_path, device=device)
    for box in results2[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(
            image2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )
        text = f"{label}: {conf:.2f}"
        cv2.putText(
            image2,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    
    # Create the comparison plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title(left_title)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title(right_title)
    plt.axis("off")
    
    if main_title:
        plt.suptitle(main_title, fontsize=16)
    
    plt.tight_layout()
    plt.show()