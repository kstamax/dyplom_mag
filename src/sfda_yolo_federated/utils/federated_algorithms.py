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
    if not model_files:
        raise ValueError("No model files provided for averaging")

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

def fednova_yolo_model(model_files, starting_model_path, output_file, local_steps=None, weights=None):
    """
    Performs FedNova (Normalized Averaging) on a list of Ultralytics YOLO model files.
    
    FedNova normalizes client updates based on the number of local steps, addressing the 
    objective inconsistency problem in federated optimization.
    
    Args:
        model_files (list of str): Paths to the YOLO model weight files.
        starting_model_path (str): Path to a base model weight file (used to create a fresh YOLO instance).
        output_file (str): Path to save the averaged model.
        local_steps (list of int, optional): Number of local training steps for each client model.
            If None, all clients are assumed to have the same number of steps.
        weights (list of float, optional): Data size weights for each model.
            If None, equal weights are used.
    """
    if not model_files:
        raise ValueError("No model files provided for averaging")
    
    # Extract global model state
    global_state = extract_state_from_yolo(starting_model_path)
    
    # Load state dictionaries
    state_dicts = []
    for f in model_files:
        state = extract_state_from_yolo(f)
        state_dicts.append(state)
    
    # Validate states are compatible
    reference_keys = set(state_dicts[0].keys())
    for i, state in enumerate(state_dicts):
        if set(state.keys()) != reference_keys:
            raise ValueError(f"Model {i} has different parameters than the first model")
    
    # Determine weights for averaging
    if weights is None:
        weights = [1.0 / len(model_files)] * len(model_files)
    else:
        if len(weights) != len(model_files):
            raise ValueError("Number of weights must match number of models")
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Determine local steps for normalization
    if local_steps is None:
        # Without specific knowledge, assume equal steps
        local_steps = [1] * len(model_files)
    elif len(local_steps) != len(model_files):
        raise ValueError("Number of local_steps entries must match number of models")
    
    # Calculate the normalization coefficient tau_eff (effective tau)
    # This is the key difference in FedNova
    a_values = [weights[i] * local_steps[i] for i in range(len(weights))]
    sum_a = sum(a_values)
    
    # To prevent division by zero
    if sum_a == 0:
        raise ValueError("Sum of normalized coefficients is zero")

    normalized_weights = [a_i / sum_a for a_i in a_values]

    print(f"FedNova normalization: Original weights: {weights}")
    print(f"FedNova normalization: Local steps: {local_steps}")
    print(f"FedNova normalization: Normalized weights: {normalized_weights}")

    # Perform normalized averaging
    avg_state = {}
    for key in reference_keys:
        # Get original dtype
        orig_dtype = state_dicts[0][key].dtype

        # Get global parameter
        w_global = global_state[key].float()

        # Compute normalized update
        update = torch.zeros_like(w_global)
        for i, sd in enumerate(state_dicts):
            # Calculate the normalized update from this client
            # Formula: (w_i - w_global) * normalized_weight_i
            client_update = sd[key].float() - w_global
            update.add_(client_update, alpha=normalized_weights[i])

        # Apply update to global model
        result = w_global + update
        
        # Convert back to original dtype
        avg_state[key] = result.to(dtype=orig_dtype)
    
    # Load to base model and save
    try:
        base_model = YOLO(starting_model_path)
        base_model.model.load_state_dict(avg_state)
        base_model.save(output_file)
        print(f"FedNova model saved to {output_file}")
        return base_model
    except Exception as e:
        raise RuntimeError(f"Failed to save averaged model: {e}")