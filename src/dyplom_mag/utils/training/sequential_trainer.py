"""
Sequential Training Module

This module provides functionality to train multiple models on different mini-datasets.
Each dataset uses the same starting model and has its own custom YAML configuration.
"""

from __future__ import annotations

import os
import subprocess
import sys
import yaml
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(__file__)


@dataclass
class TrainingConfig:
    """Configuration for sequential training process."""

    starting_model_path: str
    output_base_path: str
    final_model_template: str = "model_after_dataset_{}.pt"
    training_kwargs: Dict[str, Any] = field(default_factory=dict)
    training_kwargs_yaml: Optional[str] = None
    is_sf_yolo: bool = False
    num_datasets: int | None = None
    parallel: int = 1  # Number of parallel training processes

    def __post_init__(self):
        """Process YAML config if provided."""
        if self.training_kwargs_yaml and os.path.exists(self.training_kwargs_yaml):
            with open(self.training_kwargs_yaml, 'r') as f:
                self.training_kwargs = yaml.safe_load(f)
        elif self.training_kwargs_yaml:
            raise FileNotFoundError(f"Training kwargs YAML file not found: {self.training_kwargs_yaml}")


def train_sequential(config: TrainingConfig):
    """
    Train multiple models on different mini-datasets, with optional parallel processing.
    Each model starts from the same initial model and uses a custom YAML config.

    Args:
        config: TrainingConfig object containing configuration parameters

    Returns:
        List of paths to saved model checkpoints
    """
    # List all mini-dataset directories in the output base path
    mini_datasets = [
        os.path.join(config.output_base_path, d)
        for d in os.listdir(config.output_base_path)
        if os.path.isdir(os.path.join(config.output_base_path, d))
    ]
    mini_datasets.sort()  # Ensure consistent ordering

    # Limit number of datasets if specified
    if config.num_datasets is not None:
        mini_datasets = mini_datasets[: config.num_datasets]

    saved_models = []
    parallel = max(1, config.parallel)  # Ensure at least 1

    # Process datasets in groups based on parallel count
    for start_idx in range(0, len(mini_datasets), parallel):
        end_idx = min(start_idx + parallel, len(mini_datasets))
        current_parallel_datasets = mini_datasets[start_idx:end_idx]

        if parallel > 1:
            print(
                f"Starting parallel training for datasets {start_idx} to {end_idx - 1} ({len(current_parallel_datasets)} concurrent processes)"
            )

        # Track processes for parallel execution
        processes = []
        temp_config_files = []

        # Start training processes for each dataset in the current parallel group
        for idx_in_group, dataset_path in enumerate(current_parallel_datasets):
            dataset_idx = start_idx + idx_in_group
            dataset_name = os.path.basename(dataset_path)
            data_yaml_path = os.path.join(dataset_path, "data.yaml")

            if not os.path.exists(data_yaml_path):
                print(f"Skipping {dataset_path} because no data.yaml was found.")
                continue

            # Create a custom YAML config for this dataset
            dataset_config = copy.deepcopy(config.training_kwargs)
            # Update the data path to point to this dataset's YAML
            dataset_config["data"] = data_yaml_path
            # Add dataset name to the configuration
            dataset_config["name"] = dataset_name

            # Create a temporary YAML file for this dataset's config
            temp_config_path = os.path.join(BASE_DIR, f"temp_config_{dataset_idx}.yaml")
            with open(temp_config_path, 'w') as f:
                yaml.dump(dataset_config, f)
            temp_config_files.append(temp_config_path)

            # Prepare output model path for this specific dataset
            output_model_path = config.final_model_template.format(dataset_idx)

            # Build training command - always use the starting model
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "training_script.py"),
                "--input_model_path",
                config.starting_model_path,
                "--output_model_path",
                output_model_path,
                "--training_kwargs_yaml",
                temp_config_path,
            ]

            # Add SF YOLO flag if needed
            if config.is_sf_yolo:
                cmd.append("--is_sf_yolo")

            print(f"Starting training for dataset {dataset_idx}: {dataset_name}")

            if parallel == 1:
                # Sequential mode - use subprocess.run
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Process results immediately
                if result.stderr:
                    print(f"Subprocess stderr for {dataset_name}:")
                    print(result.stderr)

                if result.returncode != 0:
                    print(
                        f"Error: subprocess for {dataset_name} returned exit code {result.returncode}"
                    )
                    print(f"Command: {' '.join(cmd)}")
                else:
                    print(
                        f"Successfully trained on {dataset_name}, model saved to {output_model_path}"
                    )
                    saved_models.append(output_model_path)
            else:
                # Parallel mode - use subprocess.Popen
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                processes.append((process, dataset_idx, dataset_name))

        # For parallel mode, wait for all processes to complete
        if parallel > 1 and processes:
            for process, dataset_idx, dataset_name in processes:
                stdout, stderr = process.communicate()

                # Print stderr output if any
                if stderr:
                    print(
                        f"Subprocess stderr for dataset {dataset_idx} ({dataset_name}):"
                    )
                    print(stderr)

                # Check if the subprocess returned an error
                if process.returncode != 0:
                    print(
                        f"Error: subprocess for dataset {dataset_idx} ({dataset_name}) returned exit code {process.returncode}"
                    )
                else:
                    output_model_path = config.final_model_template.format(dataset_idx)
                    print(
                        f"Successfully trained on dataset {dataset_idx} ({dataset_name}), model saved to {output_model_path}"
                    )
                    saved_models.append(output_model_path)

        # Clean up temporary config files
        for temp_file in temp_config_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    if parallel > 1:
        print(
            f"Training complete. Trained {len(saved_models)} models on {len(saved_models)} datasets using up to {parallel} parallel processes."
        )
    else:
        print(f"Sequential training complete. Trained {len(saved_models)} models on {len(saved_models)} datasets.")

    return saved_models