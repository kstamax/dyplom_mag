"""
Sequential Training Module

This module provides functionality to train a model on multiple mini-datasets sequentially.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

BASE_DIR = os.path.dirname(__file__)


@dataclass
class TrainingConfig:
    """Configuration for sequential training process."""

    starting_model_path: str
    output_base_path: str
    final_model_template: str = "model_after_dataset_{}.pt"
    training_kwargs: dict[str, Any] = field(default_factory=dict)
    is_sf_yolo: bool = False
    num_datasets: int | None = None
    parallel: int = 1  # Number of parallel training processes


def train_sequential(config: TrainingConfig):
    """
    Train a model on multiple mini-datasets, with configurable parallelism.
    Setting parallel=1 gives purely sequential training.

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

    # Initialize with the starting model
    current_model_path = config.starting_model_path
    saved_models = []
    parallel = max(1, config.parallel)  # Ensure at least 1

    # Process mini-datasets in groups of size 'parallel'
    for start_idx in range(0, len(mini_datasets), parallel):
        end_idx = min(start_idx + parallel, len(mini_datasets))
        current_batch = mini_datasets[start_idx:end_idx]

        if parallel > 1:
            print(
                f"Starting training for datasets {start_idx} to {end_idx - 1} ({parallel} parallel processes)"
            )

        # Keep track of processes and their corresponding dataset indices
        processes = []
        temp_output_paths = []

        # Start multiple training processes
        for batch_idx, dataset_path in enumerate(current_batch):
            dataset_idx = start_idx + batch_idx
            dataset_name = os.path.basename(dataset_path)
            data_yaml_path = os.path.join(dataset_path, "data.yaml")

            if not os.path.exists(data_yaml_path):
                print(f"Skipping {dataset_path} because no data.yaml was found.")
                continue

            # Prepare output model path for this dataset
            output_model_path = config.final_model_template.format(dataset_idx)
            temp_output_paths.append(output_model_path)

            # Build training command with kwargs
            cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "training_script.py"),
                "--input_model_path",
                current_model_path,
                "--output_model_path",
                output_model_path,
            ]

            # Add SF YOLO flag if needed
            if config.is_sf_yolo:
                cmd.append("--is_sf_yolo")

            # Add training kwargs
            for key, value in config.training_kwargs.items():
                cmd.extend(["--training_kwargs", f"{key}={value}"])

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

                # Update current model path for next iteration in sequential mode
                current_model_path = output_model_path
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

            # Use the last successfully trained model as input for the next batch
            if temp_output_paths:
                current_model_path = temp_output_paths[-1]

    if parallel > 1:
        print(
            f"Training complete. Trained on {len(saved_models)} datasets using {parallel} parallel processes."
        )
    else:
        print(f"Sequential training complete. Trained on {len(saved_models)} datasets.")

    return saved_models
