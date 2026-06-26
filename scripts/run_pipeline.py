from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import (  # noqa: E402
    CHECK_DATA_LOADER_SCRIPT,
    DEFAULT_DATA_ROOT,
    MAKE_PLOTS_SCRIPT,
    PROJECT_ROOT,
    SRC_DIR,
    TRAIN_SCRIPT,
    VALIDATE_FAIRNESS_SCRIPT,
    BaselineExperimentConfig,
    build_train_command,
    resolve_python_bin,
)
from src.modeling.reporting import get_selected_configs, run_selected_experiments  # noqa: E402
from src.privacy.reporting import get_deid_configs, run_configs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FER project pipeline from one command."
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download/copy the Kaggle dataset before running the pipeline.",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Train baseline and fixed de-identification models before reporting.",
    )
    parser.add_argument(
        "--include-light-finetuning",
        action="store_true",
        help="Also run the light fine-tuning configs. Requires --include-training.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip training runs whose metrics already exist.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps and training commands without executing them.",
    )
    return parser.parse_args()


def run_command(label: str, command: list[str], dry_run: bool = False) -> None:
    print(f"\n[{label}]", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}.")


def print_training_commands(configs: list[BaselineExperimentConfig], data_root: Path) -> None:
    python_bin = resolve_python_bin()
    for config in configs:
        command = build_train_command(
            config=config,
            train_script=TRAIN_SCRIPT,
            python_bin=python_bin,
            data_root=data_root,
        )
        print(subprocess.list2cmdline(command))


def run_training(args: argparse.Namespace) -> None:
    baseline_configs = get_selected_configs(
        run_clean_baselines=True,
        run_light_finetuning=args.include_light_finetuning,
    )
    deid_configs = get_deid_configs()

    if args.dry_run:
        print("\n[Training commands: baselines]", flush=True)
        print_training_commands(baseline_configs, args.data_root)
        print("\n[Training commands: fixed de-identification]", flush=True)
        print_training_commands(deid_configs, args.data_root)
        return

    run_selected_experiments(
        configs=baseline_configs,
        train_script=TRAIN_SCRIPT,
        python_bin=resolve_python_bin(),
        data_root=args.data_root,
        skip_completed=args.skip_completed,
    )
    run_configs(
        configs=deid_configs,
        stage_name="fixed de-id",
        train_script=TRAIN_SCRIPT,
        python_bin=resolve_python_bin(),
        data_root=args.data_root,
        skip_completed=args.skip_completed,
    )


def main() -> None:
    args = parse_args()
    python_bin = resolve_python_bin()
    start_time = time.time()

    if args.include_light_finetuning and not args.include_training:
        raise ValueError("--include-light-finetuning requires --include-training.")

    if args.download_data:
        run_command(
            "Download dataset",
            [str(python_bin), str(SRC_DIR / "data" / "download_dataset.py")],
            dry_run=args.dry_run,
        )

    run_command(
        "DataLoader sanity check",
        [str(python_bin), str(CHECK_DATA_LOADER_SCRIPT), "--data-root", str(args.data_root)],
        dry_run=args.dry_run,
    )

    if args.include_training:
        run_training(args)
    else:
        print(
            "\n[Training]\n"
            "Skipped. Use --include-training to train/retrain baselines and fixed de-id models.",
            flush=True,
        )

    run_command(
        "Regenerate final tables and plots",
        [str(python_bin), str(MAKE_PLOTS_SCRIPT), "--data-root", str(args.data_root)],
        dry_run=args.dry_run,
    )
    run_command(
        "Validate comparison fairness",
        [str(python_bin), str(VALIDATE_FAIRNESS_SCRIPT), "--data-root", str(args.data_root)],
        dry_run=args.dry_run,
    )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nPipeline finished in {elapsed_minutes:.2f} minutes.", flush=True)


if __name__ == "__main__":
    main()
