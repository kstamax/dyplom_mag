import pandas as pd
import os

def validate_dataset(model, data_yaml_path, device="cuda"):
    results = model.val(data=data_yaml_path, split="test", device=device)
    print(results.results_dict)
    return results.results_dict


def validate_all_datasets(
    model,
    base_path="/content",
    datasets=["adit", "malam", "morning", "night", "noon", "pagi", "rain"],
    device="cuda",
    split="test"
):
    # Define dataset names and construct corresponding data.yaml paths.

    results_list = []

    for ds in datasets:
        data_yaml = os.path.join(base_path, ds, "data.yaml")
        print(f"Validating dataset: {ds} from {data_yaml}")
        try:
            results = model.val(data=data_yaml, split=split, device=device)
            metrics = results.results_dict
            results_list.append(
                {
                    "Dataset": ds,
                    "Precision": metrics.get("metrics/precision(B)", None),
                    "Recall": metrics.get("metrics/recall(B)", None),
                    "mAP50": metrics.get("metrics/mAP50(B)", None),
                    "mAP50-95": metrics.get("metrics/mAP50-95(B)", None),
                    "Fitness": metrics.get("fitness", None),
                }
            )
        except Exception as e:
            print(f"Error validating {ds}: {e}")
            results_list.append(
                {
                    "Dataset": ds,
                    "Precision": None,
                    "Recall": None,
                    "mAP50": None,
                    "mAP50-95": None,
                    "Fitness": None,
                }
            )

    # Create a DataFrame from the collected metrics.
    df = pd.DataFrame(results_list)

    # Print the DataFrame as a nice table.
    print("\nValidation Metrics:")
    print(df.to_string(index=False))
