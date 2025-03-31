import torch
from ultralytics import YOLO


def extract_state_from_yolo(file_path):
    """
    Extract the state dictionary from an Ultralytics YOLO weight file.

    The file may be:
      - A dict containing a "model" key, where the value is either a state dict or a YOLO model instance.
      - A dict that is already a state dict.
      - A YOLO model instance.
    """
    loaded = torch.load(file_path, map_location="cpu")
    if isinstance(loaded, dict):
        if "model" in loaded:
            model_obj = loaded["model"]
            if isinstance(model_obj, dict):
                return model_obj
            else:
                return model_obj.state_dict()
        else:
            return loaded
    else:
        return loaded.model.state_dict()


def fedavg_yolo_model(model_files, starting_model_path, output_file):
    """
    Performs Federated Averaging on a list of Ultralytics YOLO model files.

    Args:
        model_files (list of str): Paths to the YOLO model weight files.
        starting_model_path (str): Path to a base model weight file (used to create a fresh YOLO instance).
        output_file (str): Path to save the averaged model.
    """
    state_dicts = []
    for f in model_files:
        state = extract_state_from_yolo(f)
        state_dicts.append(state)

    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = sum([sd[key].float() for sd in state_dicts]) / len(state_dicts)

    base_model = YOLO(starting_model_path)

    base_model.model.load_state_dict(avg_state)

    base_model.save(output_file)
    print(f"Averaged model saved to {output_file}")
