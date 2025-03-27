import copy
import os
from ultralytics.utils.torch_utils import ModelEMA, de_parallel
import math
from dyplom_mag.utils.validate_model import validate_dataset
from dyplom_mag.mean_teacher_training.pseudo_dataset import prepare_pseudo_dataset

BASE_DIR = os.path.dirname(__file__)


class MyModelEMA(ModelEMA):
    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)
            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    msd_k = msd[k].detach().to(v.device)
                    v.mul_(d)
                    v.add_((1 - d) * msd_k)


def update_student_with_SSM(student_model, teacher_model, ssm_alpha):
    """
    Update student model parameters via a simple teacher-student mixing.
    """
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()
    for name, param in student_state.items():
        if name in teacher_state:
            teacher_param = teacher_state[name].to(
                param.device
            )  # ensure both are on the same device
            param.data.copy_(
                (1.0 - ssm_alpha) * param.data + ssm_alpha * teacher_param.data
            )
    print("Student model updated via SSM.")


def update_teacher_with_EMA(teacher_model, ema_obj):
    """
    Update teacher model with the EMA (exponential moving average) weights.
    """
    teacher_model.load_state_dict(ema_obj.ema.state_dict(), strict=False)
    print("Teacher model updated with EMA.")


def iterative_teacher_student_training(
    data_dir,
    teacher_model,
    total_epochs,
    conf_threshold=0.1,
    ssm_alpha=0.5,
    ema_decay=0.999,
    base_lr=0.01,
    pseudo_dataset_dir="",
    epochs=3,
    device="cuda",
    optimizer="SGD",
    batch=0.7,
    cache=False,
    imgsz=640,
):
    if not pseudo_dataset_dir:
        pseudo_dataset_dir = os.path.join(BASE_DIR, "pseudo_dataset")

    # For this example, assume images are in data_dir/valid/images
    image_folder = os.path.abspath(os.path.join(data_dir, "valid", "images"))
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(img_extensions)
    ]
    print(f"Found {len(image_files)} images in {image_folder}")

    # Ensure both teacher and student models are on the GPU.
    teacher_model = teacher_model.to(device)
    student_model = copy.deepcopy(teacher_model).to(device)

    # Initialize our custom EMA with the student model.
    ema = MyModelEMA(student_model, decay=ema_decay, tau=2000)
    first_metrics = None
    last_metrics = None
    metrics_list = []
    # Main loop over epochs.
    for epoch in range(total_epochs):
        print(f"--- Epoch {epoch + 1} ---")

        # Only apply SSM after certain intervals or with decreasing strength
        if epoch > 0 and epoch % 2 == 0:  # Apply every other epoch
            ssm_alpha_decayed = ssm_alpha * (
                1 - epoch / total_epochs
            )  # Decrease over time
            update_student_with_SSM(student_model, teacher_model, ssm_alpha_decayed)

        # Prepare pseudo dataset. Typically, you want the teacher in evaluation mode.
        teacher_model.model.eval()
        prepare_pseudo_dataset(data_dir, teacher_model, conf_threshold=conf_threshold)

        # (c) Train student on the pseudo-labeled dataset for one epoch.
        print("Training student on pseudo-labeled dataset...")
        current_lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        student_model.train(
            data=pseudo_dataset_dir,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            cache=cache,
            batch=batch,
            lr0=current_lr,
            optimizer=optimizer,
        )

        # (d) Update EMA with the student model.
        # Optionally, ensure student state_dict tensors are on CUDA (if not already).
        original_state_dict = student_model.state_dict
        student_model.state_dict = lambda: {
            k: v.to(device) for k, v in original_state_dict().items()
        }
        ema.update(student_model)
        student_model.state_dict = original_state_dict  # restore original method

        # (e) Update teacher model with EMA weights.
        teacher_model.model.train()  # set teacher back to training mode if needed
        update_teacher_with_EMA(teacher_model, ema)

        print(f"Epoch {epoch + 1} completed.\n")
        metrics_list.append(
            validate_dataset(
                copy.deepcopy(teacher_model), os.path.join(data_dir, "data.yaml"), device=device
            )
        )
        if epoch == 0:
            first_metrics = metrics_list[-1]

        if epoch == total_epochs - 1:
            last_metrics = metrics_list[-1]

    print("Iterative teacher-student training completed.")
    print(f"First metrics: {first_metrics}")
    print(f"Last metrics: {last_metrics}")
    return teacher_model, student_model, metrics_list
