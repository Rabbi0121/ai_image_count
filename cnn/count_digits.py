from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DigitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FolderDigitsDataset(Dataset):
    def __init__(self, files: list[Path], transform: transforms.Compose) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        fp = self.files[idx]
        img = Image.open(fp).convert("L")
        return self.transform(img), fp.name


@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    mnist_root: Path
    checkpoint_dir: Path
    output_json: Path
    output_csv: Path
    seeds: list[int]
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    force_retrain: bool
    recursive: bool
    extensions: set[str]
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count handwritten digits in image files (0..9) using a small CNN ensemble."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder with digit images. If omitted, auto-detects a nearby 'digits' folder.",
    )
    parser.add_argument(
        "--mnist-root",
        type=Path,
        default=None,
        help="Where MNIST is stored/downloaded. Default: <script_dir>/.mnist",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Where trained model checkpoints are stored. Default: <script_dir>/.checkpoints/digit_counter",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 13, 21, 42, 84],
        help="Random seeds for ensemble models.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per seed.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints and retrain all models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write JSON output. Default: <script_dir>/result.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write per-file predictions CSV. Default: <script_dir>/predictions.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp",
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device. Default: auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / total if total else 0.0


def train_or_load_model(
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    mnist_root: Path,
    checkpoint_dir: Path,
    force_retrain: bool,
    device: torch.device,
) -> tuple[DigitNet, float]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_dir / f"digitnet_seed{seed}_e{epochs}.pt"

    train_tf = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=8,
                fill=0,
            ),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])

    train_ds = datasets.MNIST(root=str(mnist_root), train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(root=str(mnist_root), train=False, download=True, transform=eval_tf)
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_gen,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=num_workers
    )

    set_seed(seed)
    model = DigitNet().to(device)
    if ckpt.exists() and not force_retrain:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        test_acc = evaluate(model, test_loader, device)
        return model, test_acc

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), ckpt)
    test_acc = evaluate(model, test_loader, device)
    return model, test_acc


def list_image_files(input_dir: Path, valid_suffixes: set[str], recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in valid_suffixes]
    return sorted(files)


def infer_logits(
    model: nn.Module,
    files: list[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    ds = FolderDigitsDataset(files, transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_logits: list[torch.Tensor] = []
    names: list[str] = []
    model.eval()
    with torch.no_grad():
        for x, batch_names in loader:
            x = x.to(device)
            logits = model(x).cpu()
            all_logits.append(logits)
            names.extend(batch_names)
    return torch.cat(all_logits, dim=0), names


def bincount10(values: Iterable[int]) -> list[int]:
    arr = np.array(list(values), dtype=np.int64)
    return np.bincount(arr, minlength=10).astype(int).tolist()


def resolve_input_dir(input_dir: Path | None, script_dir: Path) -> Path:
    if input_dir is not None:
        return input_dir.expanduser().resolve()

    candidates = [
        script_dir / "digits",
        script_dir.parent / "digits",
        Path.cwd() / "digits",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not auto-detect the digits folder. Checked:\n"
        f"{searched}\n"
        "Pass --input-dir explicitly."
    )


def parse_extensions(raw_extensions: str) -> set[str]:
    exts: set[str] = set()
    for part in raw_extensions.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    if not exts:
        raise ValueError("No valid --extensions were provided.")
    return exts


def resolve_config(args: argparse.Namespace) -> RunConfig:
    script_dir = Path(__file__).resolve().parent
    input_dir = resolve_input_dir(args.input_dir, script_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")

    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")

    return RunConfig(
        input_dir=input_dir,
        mnist_root=(args.mnist_root or (script_dir / ".mnist")).expanduser().resolve(),
        checkpoint_dir=(args.checkpoint_dir or (script_dir / ".checkpoints/digit_counter")).expanduser().resolve(),
        output_json=(args.output_json or (script_dir / "result.json")).expanduser().resolve(),
        output_csv=(args.output_csv or (script_dir / "predictions.csv")).expanduser().resolve(),
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        force_retrain=args.force_retrain,
        recursive=args.recursive,
        extensions=parse_extensions(args.extensions),
        device=get_device(args.device),
    )


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    files = list_image_files(
        input_dir=config.input_dir,
        valid_suffixes=config.extensions,
        recursive=config.recursive,
    )
    if not files:
        raise RuntimeError(f"No image files found in: {config.input_dir}")

    print(f"Input directory: {config.input_dir}")
    print(f"Recursive scan: {config.recursive}")
    print(f"Extensions: {', '.join(sorted(config.extensions))}")
    print(f"Device: {config.device}")
    print(f"MNIST cache: {config.mnist_root}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"JSON output: {config.output_json}")
    print(f"CSV output: {config.output_csv}")

    models: list[DigitNet] = []
    model_accs: list[float] = []
    for seed in config.seeds:
        model, acc = train_or_load_model(
            seed=seed,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            mnist_root=config.mnist_root,
            checkpoint_dir=config.checkpoint_dir,
            force_retrain=config.force_retrain,
            device=config.device,
        )
        models.append(model)
        model_accs.append(acc)
        print(f"Seed {seed}: MNIST test accuracy = {acc:.4f}")

    logits_sum = None
    names: list[str] = []
    for i, model in enumerate(models):
        logits, batch_names = infer_logits(
            model=model,
            files=files,
            batch_size=max(256, config.batch_size),
            num_workers=config.num_workers,
            device=config.device,
        )
        if i == 0:
            names = batch_names
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    assert logits_sum is not None
    ensemble_logits = logits_sum / len(models)
    probs = torch.softmax(ensemble_logits, dim=1)
    preds = probs.argmax(dim=1).numpy()
    confidences = probs.max(dim=1).values.numpy()

    counts = bincount10(preds)
    output = {
        "counts": counts,
        "answer_array_0_to_9": counts,
        "label_counts": {str(i): counts[i] for i in range(10)},
        "total_files": len(files),
        "sum_counts": int(sum(counts)),
        "seeds": config.seeds,
        "epochs": config.epochs,
        "mnist_test_accuracies": [round(float(a), 6) for a in model_accs],
        "confidence": {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "p01": float(np.percentile(confidences, 1)),
            "p05": float(np.percentile(confidences, 5)),
        },
    }

    print(json.dumps(output, indent=2))

    try:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write JSON output to {config.output_json}: {exc}", file=sys.stderr)

    try:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with config.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prediction", "confidence"])
            for name, pred, conf in zip(names, preds.tolist(), confidences.tolist()):
                writer.writerow([name, int(pred), float(conf)])
    except OSError as exc:
        print(f"Warning: could not write CSV output to {config.output_csv}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
