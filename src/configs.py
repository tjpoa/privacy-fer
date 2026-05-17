from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_DATASET_NAME = "balanced-raf-db-dataset-7575-grayscale"
DEFAULT_DATA_ROOT = RAW_DATA_DIR / DEFAULT_DATASET_NAME

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
BASELINE_PLOT_PATH = RESULTS_PLOTS_DIR / "baseline_metrics.png"

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

SUPPORTED_MODELS = ("resnet18", "mobilenet_v3_large", "swin_t")
SUPPORTED_WEIGHTS = ("pretrained", "random")
SUPPORTED_PRIVACY_MODES = ("none", "blur", "edges", "noise")

CLASS_NAMES = (
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_python_bin() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def format_intensity(value: float | int) -> str:
    return str(float(value)).replace(".", "p")


def normalize_run_suffix(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def build_run_name(
    model: str,
    privacy_mode: str,
    privacy_intensity: float | int,
    run_suffix: str | None = None,
) -> str:
    base_name = f"{model.strip().lower()}_{privacy_mode}_{format_intensity(privacy_intensity)}"
    normalized_suffix = normalize_run_suffix(run_suffix)
    if normalized_suffix:
        return f"{base_name}_{normalized_suffix}"
    return base_name


def ensure_results_dirs() -> tuple[Path, Path]:
    RESULTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_MODELS_DIR, RESULTS_PLOTS_DIR


@dataclass
class BaselineExperimentConfig:
    model: str = "resnet18"
    weights: str = "pretrained"
    privacy_mode: str = "none"
    privacy_intensity: float = 0.0
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    seed: int = 42
    max_samples_per_split: int | None = None
    smoke_test: bool = False
    run_suffix: str | None = None
    log_interval: int = 50

    @classmethod
    def smoke_test_config(cls, **overrides) -> "BaselineExperimentConfig":
        config = cls(
            epochs=1,
            batch_size=8,
            max_samples_per_split=64,
            smoke_test=True,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @classmethod
    def safe_gpu_config(cls, **overrides) -> "BaselineExperimentConfig":
        config = cls(
            model="mobilenet_v3_large",
            weights="pretrained",
            privacy_mode="none",
            privacy_intensity=0.0,
            epochs=5,
            batch_size=32,
            image_size=160,
            num_workers=0,
            smoke_test=False,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @classmethod
    def original_config(cls, **overrides) -> "BaselineExperimentConfig":
        config = cls(
            model="resnet18",
            weights="pretrained",
            privacy_mode="none",
            privacy_intensity=0.0,
            epochs=10,
            batch_size=32,
            image_size=224,
            num_workers=0,
            smoke_test=False,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @classmethod
    def gpu_tuned_config(cls, **overrides) -> "BaselineExperimentConfig":
        config = cls(
            model="resnet18",
            weights="pretrained",
            privacy_mode="none",
            privacy_intensity=0.0,
            epochs=10,
            batch_size=64,
            image_size=224,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            smoke_test=False,
            run_suffix="gpu_tuned",
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @classmethod
    def swin_baseline_config(cls, **overrides) -> "BaselineExperimentConfig":
        config = cls(
            model="swin_t",
            weights="pretrained",
            privacy_mode="none",
            privacy_intensity=0.0,
            epochs=10,
            batch_size=32,
            image_size=224,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            smoke_test=False,
            run_suffix="swin_baseline",
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @property
    def run_name(self) -> str:
        return build_run_name(
            self.model,
            self.privacy_mode,
            self.privacy_intensity,
            self.run_suffix,
        )

    @property
    def plot_path(self) -> Path:
        return RESULTS_PLOTS_DIR / f"{self.run_name}_metrics.png"

    @property
    def metrics_path(self) -> Path:
        return RESULTS_MODELS_DIR / f"{self.run_name}_metrics.json"

    @property
    def report_path(self) -> Path:
        return RESULTS_MODELS_DIR / f"{self.run_name}_classification_report.txt"

    @property
    def checkpoint_path(self) -> Path:
        return RESULTS_MODELS_DIR / f"{self.run_name}_best.pt"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["run_name"] = self.run_name
        payload["plot_path"] = str(self.plot_path)
        payload["metrics_path"] = str(self.metrics_path)
        payload["report_path"] = str(self.report_path)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


def build_train_command(
    config: BaselineExperimentConfig,
    train_script: Path,
    python_bin: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[str]:
    python_bin = python_bin or resolve_python_bin()

    command = [
        str(python_bin),
        str(train_script),
        "--model",
        config.model,
        "--weights",
        config.weights,
        "--data-root",
        str(data_root),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--num-workers",
        str(config.num_workers),
        "--image-size",
        str(config.image_size),
        "--privacy-mode",
        config.privacy_mode,
        "--privacy-intensity",
        str(config.privacy_intensity),
        "--seed",
        str(config.seed),
        "--plot-path",
        str(config.plot_path),
        "--log-interval",
        str(config.log_interval),
    ]

    if config.pin_memory is True:
        command.append("--pin-memory")
    elif config.pin_memory is False:
        command.append("--no-pin-memory")

    if config.persistent_workers:
        command.append("--persistent-workers")

    if config.prefetch_factor is not None:
        command.extend(["--prefetch-factor", str(config.prefetch_factor)])

    if config.run_suffix:
        command.extend(["--run-suffix", config.run_suffix])

    if config.max_samples_per_split is not None:
        command.extend(["--max-samples-per-split", str(config.max_samples_per_split)])

    return command
