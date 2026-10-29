"""
TRUST-X Complete Experiment Suite Orchestrator
Executes all 12 model-language experiments in structured sequence:
1. Telugu (mBERT, IndicBERT, XLM-R)
2. Tamil (mBERT, IndicBERT, XLM-R)
3. Malayalam (mBERT, IndicBERT, XLM-R)
4. Hindi (mBERT, IndicBERT, XLM-R)

Supports running single languages or models via CLI flags.
Features:
- Automatic skip logic for already-completed experiments
- Per-experiment GPU memory cleanup (gc.collect + torch.cuda.empty_cache)
- CLI filtering by --language and --model
"""

import os
import sys
import argparse
import json
import gc
import yaml
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.training.trainer import train_and_evaluate

LANGUAGES = ["telugu", "tamil", "malayalam", "hindi"]
MODELS = ["mbert", "indicbert", "xlmr"]

def cleanup_cuda():
    """Clean up memory across CPU and CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="TRUST-X Master Experiment Orchestrator")
    parser.add_argument("--language", type=str, choices=LANGUAGES, default=None, help="Filter by specific language")
    parser.add_argument("--model", type=str, choices=MODELS, default=None, help="Filter by specific model")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--no_shap", action="store_true", help="Skip SHAP explainability")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun even if experiment already completed")

    args = parser.parse_args()

    active_langs = [args.language] if args.language else LANGUAGES
    active_models = [args.model] if args.model else MODELS

    print("=" * 80)
    print("TRUST-X MASTER EXPERIMENT SUITE")
    print(f"Target Languages: {active_langs}")
    print(f"Target Models:    {active_models}")
    print(f"Total Runs:       {len(active_langs) * len(active_models)}")
    print("=" * 80)

    configs_dir = os.path.join(root_dir, "configs")
    successful_runs = []
    skipped_runs = []
    failed_runs = []

    for lang in active_langs:
        for model in active_models:
            cfg_name = f"{lang}_{model}.yaml"
            cfg_path = os.path.join(configs_dir, cfg_name)

            if not os.path.exists(cfg_path):
                print(f"[-] Config not found: {cfg_path}")
                failed_runs.append((lang, model, "Config not found"))
                continue

            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if args.epochs is not None:
                config["epochs"] = args.epochs
            if args.batch_size is not None:
                config["batch_size"] = args.batch_size
            if args.no_shap:
                config["run_shap"] = False

            config["data_dir"] = os.path.join(root_dir, config["data_dir"])
            config["output_dir"] = os.path.join(root_dir, config["output_dir"])

            experiment_id = config.get("experiment_id", f"{lang}_{model}")

            # 1. Check for already-completed experiment
            results_json_path = os.path.join(config["output_dir"], f"results_{experiment_id}.json")
            if not args.force_rerun and os.path.exists(results_json_path):
                try:
                    with open(results_json_path, "r", encoding="utf-8") as rf:
                        prior_res = json.load(rf)
                    if prior_res.get("status") == "COMPLETED":
                        print(f"[SKIP] Experiment {experiment_id} already COMPLETED. Skipping.")
                        skipped_runs.append(experiment_id)
                        continue
                except Exception as json_err:
                    print(f"[!] Warning reading existing results JSON for {experiment_id}: {json_err}")

            print(f"\n>>> LAUNCHING RUN: {experiment_id} <<<")
            try:
                exp_summary = train_and_evaluate(config)
                successful_runs.append(experiment_id)
            except Exception as e:
                print(f"\n[!] EXPERIMENT FAILED: {experiment_id} - {str(e)}")
                failed_runs.append((lang, model, str(e)))
            finally:
                # 2. CUDA Memory Cleanup after every run attempt (success or failure)
                cleanup_cuda()

    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE RUN SUMMARY")
    print("=" * 80)
    print(f"Total Targets:    {len(active_langs) * len(active_models)}")
    print(f"Total Completed:  {len(successful_runs)}")
    print(f"Total Skipped:    {len(skipped_runs)}")
    print(f"Total Failed:     {len(failed_runs)}")
    for exp in successful_runs:
        print(f"  [+] COMPLETED: {exp}")
    for exp in skipped_runs:
        print(f"  [*] SKIPPED:   {exp} (already COMPLETED)")
    for lang, model, err in failed_runs:
        print(f"  [-] FAILED:    {lang}_{model} - {err}")
    print("=" * 80)

if __name__ == "__main__":
    main()
