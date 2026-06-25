from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.configs import DEFAULT_DATA_ROOT, RESULTS_TABLES_DIR, ensure_results_dirs
from src.data_loader import RAFDataset


REPRESENTATIVE_TRANSFORMS = [
    ("baseline_clean", "none", 0.0),
    ("crop_context_removal", "crop", 0.75),
    ("blur", "blur", 3.0),
    ("mosaic", "mosaic", 8.0),
    ("canny", "edges", 0.0),
    ("noise", "noise", 100.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether FER model comparisons use fair dataset splits and transformation protocol."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=RESULTS_TABLES_DIR,
        help="Directory containing final tables and where validation reports are saved.",
    )
    return parser.parse_args()


def dataset_index(data_root: Path, split: str, mode: str, intensity: float) -> pd.DataFrame:
    dataset = RAFDataset(
        root_dir=data_root,
        split=split,
        mode=mode,
        intensity=intensity,
        transform=None,
        grayscale=True,
        return_metadata=True,
    )
    return pd.DataFrame(
        [
            {
                "image_path": str(record.image_path),
                "file_name": record.file_name,
                "split": record.split,
                "class_name": record.class_name,
                "target": record.target,
            }
            for record in dataset.records
        ]
    )


def check_dataset_splits(data_root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        baseline = dataset_index(data_root, split, "none", 0.0)
        baseline_signature = baseline[["image_path", "target", "class_name"]].reset_index(drop=True)
        class_distribution = baseline["class_name"].value_counts().sort_index().to_dict()

        checks.append(
            {
                "check": f"{split}_baseline_distribution",
                "status": "PASS",
                "evidence": json.dumps(
                    {
                        "n_samples": int(len(baseline)),
                        "class_distribution": class_distribution,
                    },
                    sort_keys=True,
                ),
            }
        )

        for transform_label, mode, intensity in REPRESENTATIVE_TRANSFORMS:
            transformed = dataset_index(data_root, split, mode, intensity)
            transformed_signature = transformed[["image_path", "target", "class_name"]].reset_index(drop=True)
            same_samples = baseline_signature.equals(transformed_signature)
            checks.append(
                {
                    "check": f"{split}_{transform_label}_same_samples_as_baseline",
                    "status": "PASS" if same_samples else "FAIL",
                    "evidence": (
                        f"mode={mode}, intensity={intensity}, "
                        f"baseline_n={len(baseline)}, transformed_n={len(transformed)}"
                    ),
                }
            )
    return checks


def check_results_summary(tables_dir: Path) -> list[dict[str, object]]:
    summary_path = tables_dir / "results_summary.csv"
    if not summary_path.exists():
        return [
            {
                "check": "results_summary_exists",
                "status": "FAIL",
                "evidence": f"Missing {summary_path}",
            }
        ]

    df = pd.read_csv(summary_path)
    required_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "transformation",
        "privacy_mode",
        "privacy_intensity",
        "macro_f1_drop_vs_clean",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    checks = [
        {
            "check": "results_summary_exists",
            "status": "PASS",
            "evidence": f"{summary_path} rows={len(df)}",
        },
        {
            "check": "results_summary_required_columns",
            "status": "PASS" if not missing_columns else "FAIL",
            "evidence": "missing=" + ",".join(missing_columns) if missing_columns else "all required columns present",
        },
    ]

    expected_transformations = {
        "baseline_clean",
        "crop_context_removal",
        "blur",
        "mosaic",
        "canny",
        "noise",
    }
    observed_transformations = set(df["transformation"].dropna().unique())
    missing_transformations = sorted(expected_transformations - observed_transformations)
    checks.append(
        {
            "check": "results_summary_transformation_coverage",
            "status": "PASS" if not missing_transformations else "WARN",
            "evidence": (
                "observed="
                + ",".join(sorted(observed_transformations))
                + "; missing="
                + ",".join(missing_transformations)
            ),
        }
    )
    return checks


def check_test_tuning_protocol(tables_dir: Path) -> list[dict[str, object]]:
    proxy_path = Path("results/deid_proxy_selection.csv")
    fixed_path = Path("results/deid_fixed_comparison.csv")
    checks: list[dict[str, object]] = []

    if proxy_path.exists():
        proxy_df = pd.read_csv(proxy_path)
        has_validation_selection_columns = {
            "val_f1_macro",
            "val_f1_drop",
            "within_drop_limit",
            "privacy_score",
        }.issubset(proxy_df.columns)
        checks.append(
            {
                "check": "privacy_parameter_selection_has_validation_columns",
                "status": "PASS" if has_validation_selection_columns else "WARN",
                "evidence": f"{proxy_path}; columns={','.join(proxy_df.columns)}",
            }
        )
    else:
        checks.append(
            {
                "check": "privacy_parameter_selection_has_validation_columns",
                "status": "WARN",
                "evidence": f"{proxy_path} not found; cannot verify selection table",
            }
        )

    if fixed_path.exists():
        fixed_df = pd.read_csv(fixed_path)
        has_val_and_test = {
            "val_f1_macro",
            "test_f1_macro",
            "original_f1_macro",
            "delta_f1_macro",
        }.issubset(fixed_df.columns)
        checks.append(
            {
                "check": "final_deid_table_separates_validation_and_test_metrics",
                "status": "PASS" if has_val_and_test else "WARN",
                "evidence": f"{fixed_path}; rows={len(fixed_df)}",
            }
        )
    else:
        checks.append(
            {
                "check": "final_deid_table_separates_validation_and_test_metrics",
                "status": "WARN",
                "evidence": f"{fixed_path} not found",
            }
        )

    checks.append(
        {
            "check": "test_set_not_used_for_tuning",
            "status": "WARN",
            "evidence": (
                "Code-level evidence supports validation-based selection: train_baseline.py selects "
                "best_epoch using validation macro-F1 and deid_proxy_selection.csv contains validation "
                "selection columns. Human notebook decisions cannot be fully proven from artifacts alone; "
                "report should state that parameter choices are based on validation metrics and test is reserved "
                "for final reporting."
            ),
        }
    )
    return checks


def write_markdown_report(checks: list[dict[str, object]], output_path: Path) -> None:
    status_counts = pd.Series([check["status"] for check in checks]).value_counts().to_dict()
    lines = [
        "# Fairness Validation Report",
        "",
        "This report checks whether the comparison protocol uses consistent splits, consistent transformation application, and a validation-first tuning workflow.",
        "",
        "## Summary",
        "",
    ]
    for status in ("PASS", "WARN", "FAIL"):
        lines.append(f"- {status}: {status_counts.get(status, 0)}")
    lines.extend(["", "## Checks", ""])
    for check in checks:
        lines.append(f"- **{check['status']}** `{check['check']}`: {check['evidence']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    checks.extend(check_dataset_splits(args.data_root))
    checks.extend(check_results_summary(args.tables_dir))
    checks.extend(check_test_tuning_protocol(args.tables_dir))

    checks_df = pd.DataFrame(checks)
    csv_path = args.tables_dir / "fairness_validation.csv"
    json_path = args.tables_dir / "fairness_validation_report.json"
    md_path = args.tables_dir / "fairness_validation_report.md"

    checks_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    write_markdown_report(checks, md_path)

    print("Fairness validation finished.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(checks_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
