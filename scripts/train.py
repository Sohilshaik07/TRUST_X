"""
TRUST-X Training CLI Entrypoint
Usage:
    python scripts/train.py --config configs/telugu_mbert.yaml
    python scripts/train.py --config configs/tamil_xlmr.yaml --epochs 4 --batch_size 32
"""

import os
import sys
import argparse
import yaml

# Ensure project root is in path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.training.trainer import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="TRUST-X Model Training & Evaluation Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no_shap", action="store_true", help="Disable SHAP explainability run to save time")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        # Check relative to root_dir
        rel_path = os.path.join(root_dir, args.config)
        if os.path.exists(rel_path):
            config_path = rel_path
        else:
            raise FileNotFoundError(f"Config file not found at '{args.config}' or '{rel_path}'")
    else:
        config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides if provided
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.no_shap:
        config["run_shap"] = False

    # Ensure paths are resolved relative to root_dir
    if not os.path.isabs(config["data_dir"]):
        config["data_dir"] = os.path.join(root_dir, config["data_dir"])
    if not os.path.isabs(config["output_dir"]):
        config["output_dir"] = os.path.join(root_dir, config["output_dir"])

    train_and_evaluate(config)

if __name__ == "__main__":
    main()
