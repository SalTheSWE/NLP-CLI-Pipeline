from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any

import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("outputs")
    visualizations: str = "visualizations"
    processed_csvs: str = "processed_csvs"
    generated_csvs: str = "generated_csvs"
    embeddings: str = "embeddings"
    models: str = "models"
    reports: str = "reports"
    ir_index: str = "ir_index"

    csv_encoding: str = "utf-8-sig"
    fig_dpi: int = 200
    fig_ext: str = "png"


CFG = OutputConfig()


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def bucket_dir(bucket: str) -> Path:
    """
    bucket must be one of:
    visualizations, processed_csvs, generated_csvs, embeddings, models, reports, ir_index
    """
    if not hasattr(CFG, bucket):
        raise ValueError(f"Unknown bucket '{bucket}'.")
    return _ensure_dir(CFG.root / getattr(CFG, bucket))


def resolve_output_path(output, *, bucket, base_name, ext, add_timestamp=True):
    # If user provides output -> use it, but if it's only a filename, put it in the bucket folder
    if output:
        p = Path(output)

        # "cleaned.csv" (no parent directory provided)
        if p.parent == Path("."):
            out_dir = bucket_dir(bucket)
            return out_dir / p.name

        # "outputs/processed_csvs/cleaned.csv" or "some/dir/file.csv"
        _ensure_dir(p.parent)
        return p

    # If output not provided -> auto-generate in bucket dir
    out_dir = bucket_dir(bucket)
    filename = f"{base_name}{'_' + ts() if add_timestamp else ''}.{ext}"
    return out_dir / filename



def save_csv(
    df: pd.DataFrame,
    output: Optional[PathLike],
    *,
    bucket: str = "processed_csvs",
    base_name: str = "data",
    add_timestamp: bool = True,
) -> str:
    path = resolve_output_path(
        output, bucket=bucket, base_name=base_name, ext="csv", add_timestamp=add_timestamp
    )
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding=CFG.csv_encoding)
    return str(path)


def save_text(
    text: str,
    output: Optional[PathLike],
    *,
    bucket: str = "reports",
    base_name: str = "report",
    ext: str = "txt",
    add_timestamp: bool = True,
) -> str:
    path = resolve_output_path(
        output, bucket=bucket, base_name=base_name, ext=ext, add_timestamp=add_timestamp
    )
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return str(path)


def save_pickle(
    obj: Any,
    output: Optional[PathLike],
    *,
    bucket: str = "embeddings",
    base_name: str = "artifact",
    add_timestamp: bool = True,
) -> str:
    import pickle

    path = resolve_output_path(
        output, bucket=bucket, base_name=base_name, ext="pkl", add_timestamp=add_timestamp
    )
    _ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return str(path)


def save_current_figure(
    output: Optional[PathLike],
    *,
    bucket: str = "visualizations",
    base_name: str = "figure",
    add_timestamp: bool = True,
    dpi: Optional[int] = None,
) -> str:
    import matplotlib.pyplot as plt

    path = resolve_output_path(
        output, bucket=bucket, base_name=base_name, ext=CFG.fig_ext, add_timestamp=add_timestamp
    )
    _ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=(dpi or CFG.fig_dpi), bbox_inches="tight")
    return str(path)


def append_command_log(bucket: str, line: str) -> str:
    """
    Appends a line to outputs/<bucket>/commands_to_output.txt
    """
    d = bucket_dir(bucket)
    log_path = d / "commands_to_output.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
    return str(log_path)