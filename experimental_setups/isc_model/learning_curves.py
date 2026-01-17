import argparse
import os
from typing import Optional

import pandas as pd
import torch

import data
import utils
from . import model

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train_models_and_collect_learning_curves(
    num_models: int = 3,
    epochs: int = 30,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Train ISC models and collect per-epoch BCE loss values.

    Returns a DataFrame with columns ``model``, ``epoch``, and ``bce``.
    """
    if device is None:
        device = utils.set_torch_device()

    train_x, train_y = data.make_training_data(device=device)
    runs = []

    for model_idx in range(num_models):
        isc_model = model.ISCModel(device=device)
        metrics = isc_model.train(train_x, train_y, epochs=epochs, batch_size=batch_size)
        bce_metric = next((m for m in metrics if getattr(m, "name", "") == "bce"), None)
        if bce_metric is None or not getattr(bce_metric, "values", None):
            continue

        bce_values = [float(v) for v in bce_metric.values]
        runs.append(
            pd.DataFrame(
                {
                    "model": model_idx,
                    "epoch": range(len(bce_values)),
                    "bce": bce_values,
                }
            )
        )

    if not runs:
        raise RuntimeError("No learning curves collected.")

    return pd.concat(runs, ignore_index=True)


def plot_learning_curves(curves: pd.DataFrame, output_path: str) -> None:
    """
    Plot BCE learning curves for one or more trained ISC models.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting learning curves.")

    _ensure_parent_dir(output_path)

    plt.figure(figsize=(8, 5))
    for model_id, group in curves.groupby("model"):
        plt.plot(group["epoch"], group["bce"], label=f"Model {model_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("ISC model learning curves")
    plt.legend(title="Run", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISC models and plot learning curves.")
    parser.add_argument("--num-models", type=int, default=3, help="Number of ISC models to train.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train each model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used during training.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (defaults to auto-selected device).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/isc_learning_curves.csv",
        help="Where to save per-epoch loss values.",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default="figures/isc_learning_curves.png",
        help="Where to save the learning curve plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the plot (useful if matplotlib is unavailable).",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else utils.set_torch_device()

    curves = train_models_and_collect_learning_curves(
        num_models=args.num_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    if args.output_csv:
        _ensure_parent_dir(args.output_csv)
        curves.to_csv(args.output_csv, index=False)
        print(f"Saved learning curves to {args.output_csv}")

    if args.output_fig and not args.no_plot:
        if plt is None:
            print("matplotlib is not installed; skipping plot export. Install it to save figures.")
        else:
            plot_learning_curves(curves, args.output_fig)
            print(f"Saved learning curve plot to {args.output_fig}")
    elif args.no_plot:
        print("Plot export skipped by flag.")


if __name__ == "__main__":
    main()
