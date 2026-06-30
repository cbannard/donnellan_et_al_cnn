import argparse
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score
from captum.attr import IntegratedGradients
from analysis_utils import assign_top_quartile_bins, assign_bottom_quartile_bins, stratified_random_baseline


@dataclass
class Sample:
    participant: int
    x: np.ndarray
    y_bin: int
    y_raw: float


class TimeSeriesDataset(Dataset):
    def __init__(
        self, samples: Sequence[Sample], window_size: int = None, train: bool = True
    ):
        self.samples = list(samples)
        self.window_size = window_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = sample.x

        return {
            "participant": sample.participant,
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(sample.y_bin, dtype=torch.long),
        }


def collate_batch(batch: list[dict]):
    x_list = [item["x"] for item in batch]
    y = torch.stack([item["y"] for item in batch])

    lengths = torch.tensor([seq.shape[1] for seq in x_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    channels = x_list[0].shape[0]

    padded_x = torch.zeros(len(batch), channels, max_len, dtype=torch.float32)
    for i, seq in enumerate(x_list):
        padded_x[i, :, : seq.shape[1]] = seq

    return {
        "participant": [item["participant"] for item in batch],
        "x": padded_x,
        "y": y,
    }


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self.dropout(self.activation(self.conv1(x)))
        out = self.dropout(self.activation(self.conv2(out)))
        return self.activation(out + res)


class CnnModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        embed: int = 64,
        drop: float = 0.2,
        classes: int = 2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            TemporalBlock(in_channels, hidden, 5, 1, drop),
            TemporalBlock(hidden, embed, 5, 2, drop),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = torch.max(features, dim=-1).values
        return self.regressor(pooled)


def load_dataset(cdi_path: Path, time_series_dir: Path, data_age: str, target_criterion: str = "top") -> list[Sample]:
    # 1. Load targets
    df = pd.read_csv(cdi_path)
    df["us18"] = pd.to_numeric(df["us18_imp"], errors="coerce")
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce")
    valid_rows = df.dropna(subset=["participant", "us18"])
    targets = {
        int(r.participant): float(r.us18) for r in valid_rows.itertuples(index=False)
    }

    # 2. Load time-series data (filter before computing threshold)
    valid_samples = []  # list of (participant, us18, values)
    for participant, us18 in sorted(targets.items()):
        file_path = time_series_dir / f"p{participant}_{data_age}.csv"
        if not file_path.exists():
            continue

        frame = pd.read_csv(file_path, index_col=0)
        frame.index = frame.index.astype(str).str.strip().str.strip('"')

        offshot = (
            frame.loc["OFFSHOT"]
            if "OFFSHOT" in frame.index
            else pd.Series(0, index=frame.columns)
        )
        padding = (
            frame.loc["PADDING"]
            if "PADDING" in frame.index
            else pd.Series(0, index=frame.columns)
        )

        offshot = pd.to_numeric(offshot, errors="coerce").fillna(0.0)
        padding = pd.to_numeric(padding, errors="coerce").fillna(0.0)
        valid_time = ((offshot != 1.0) & (padding != 1.0)).to_numpy()

        predictor_rows = [
            lbl for lbl in frame.index if lbl not in {"OFFSHOT", "PADDING"}
        ]
        predictor_frame = frame.loc[predictor_rows, frame.columns[valid_time]]
        values = (
            predictor_frame.apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        if values.shape[1] < 25:  # Drop sequences that are too short to window
            continue

        valid_samples.append((participant, us18, values))

    # 3. Compute threshold on the actual sample (not the full CSV population)
    us18_vals = np.array([us18 for _, us18, _ in valid_samples])
    
    if target_criterion == "bottom":
        bins, _ = assign_bottom_quartile_bins(us18_vals)
    else:
        bins, _ = assign_top_quartile_bins(us18_vals)

    samples = []
    for (participant, us18, values), bin_idx in zip(valid_samples, bins):
        samples.append(Sample(participant, values, int(bin_idx), float(us18)))

    return samples


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cdi-path",
        type=Path,
        default=Path("CDI_data_w_18_withimputed.csv"),
    )
    parser.add_argument(
        "--time-series-dir",
        type=Path,
        default=Path("Donnellan_et_al_Data/Donnellan_et_al_Data/time_series_data"),
    )
    parser.add_argument("--data-age", type=str, choices=["11m", "12m"], required=True)
    parser.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="Matches the exact receptive field width of the CNN",
    )
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)  #15
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-class", type=str, choices=["top", "bottom"], default="top", help="Whether class 1 is the top 25% or bottom 25% of us18 values")
    args = parser.parse_args()

    # Load whole samples (top quartile vs rest)
    args.num_bins = 2
    all_samples = load_dataset(args.cdi_path, args.time_series_dir, args.data_age, args.target_class)
    print(f"Loaded {len(all_samples)} samples for {args.data_age} age group.")
    if len(all_samples) == 0:
        return

    loo = LeaveOneOut()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    selected_slices = []

    # Store all per-slice true targets and predicted labels to calculate F1 score at the end
    all_slice_true = []
    all_slice_pred = []

    # Store whole-input (full sequence) predictions
    all_whole_true = []
    all_whole_pred = []

    fold_pbar = tqdm(
        enumerate(loo.split(all_samples)), total=len(all_samples), desc="LOO Folds"
    )
    for fold_idx, (train_idx, test_idx) in fold_pbar:

        train_samples = [all_samples[i] for i in train_idx]
        test_sample = all_samples[test_idx[0]]

        # Standardize based on train set
        mean = np.concatenate([s.x for s in train_samples], axis=1).mean(
            axis=1, keepdims=True
        )
        std = np.concatenate([s.x for s in train_samples], axis=1).std(
            axis=1, keepdims=True
        )
        std = np.where(std < 1e-6, 1.0, std)

        train_samples = [
            replace(sample, x=(sample.x - mean) / std) for sample in train_samples
        ]

        train_loader = DataLoader(
            TimeSeriesDataset(train_samples, window_size=None, train=True),
            batch_size=16,
            shuffle=True,
            collate_fn=collate_batch,
        )

        in_channels = train_samples[0].x.shape[0]
        model = CnnModel(in_channels=in_channels, classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Calculate class weights for imbalanced training (25% class 1 vs 75% class 0)
        y_train = [s.y_bin for s in train_samples]
        counts = np.bincount(y_train)
        weights = torch.tensor(1.0 / counts, dtype=torch.float32)
        weights = weights / weights.sum()  # normalize
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

        model.train()
        for epoch in range(args.epochs):
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

            # Update the progress bar with last loss
            fold_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Sliding Window Testing
        model.eval()
        test_x = (test_sample.x - mean) / std
        target_class = test_sample.y_bin

        seq_len = test_x.shape[1]

        with torch.no_grad():
            # Whole-input prediction
            x_full = torch.from_numpy(test_x).unsqueeze(0).to(device)
            logits_full = model(x_full)
            pred_whole = logits_full.argmax(dim=-1).item()
            all_whole_true.append(target_class)
            all_whole_pred.append(pred_whole)

            participant_slices = []
            for start in range(0, seq_len - args.window_size + 1, args.stride):
                slice_data = test_x[:, start : start + args.window_size]
                x_tensor = torch.from_numpy(slice_data).unsqueeze(0).to(device)

                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=-1)
                pred_class = logits.argmax(dim=-1).item()
                prob_class1 = probs[0, 1].item()

                all_slice_true.append(target_class)
                all_slice_pred.append(pred_class)

                participant_slices.append(
                    {
                        "participant": test_sample.participant,
                        "class": target_class,
                        "start": start,
                        "prob_class1": prob_class1,
                        "slice_features": slice_data.copy(),
                    }
                )

            # Select top and bottom 20% by class-1 probability
            if participant_slices:
                n_select = max(1, int(len(participant_slices) * 0.2))

                # By probability
                sorted_slices_prob = sorted(
                    participant_slices, key=lambda s: s["prob_class1"]
                )

                # By attribution
                ig = IntegratedGradients(model)
                test_x_tensor = torch.from_numpy(test_x).unsqueeze(0).to(device)

                try:
                    with torch.enable_grad():
                        test_x_tensor.requires_grad_()
                        attributions, delta = ig.attribute(
                            test_x_tensor, target=1, return_convergence_delta=True
                        )
                    attributions = (
                        attributions.detach().squeeze(0).cpu().numpy()
                    )  # [channels, time]

                    # Calculate total attribution for each window
                    for s in participant_slices:
                        start = s["start"]
                        window_attr = attributions[:, start : start + args.window_size]
                        s["total_attribution"] = window_attr.sum()

                    sorted_slices_attr = sorted(
                        participant_slices, key=lambda s: s.get("total_attribution", 0)
                    )

                    # Tag slices with their selection method
                    for s in sorted_slices_prob[:n_select]:
                        selected_slices.append({**s, "selection": "prob_bottom"})
                    for s in sorted_slices_prob[-n_select:]:
                        selected_slices.append({**s, "selection": "prob_top"})

                    for s in sorted_slices_attr[:n_select]:
                        selected_slices.append({**s, "selection": "attr_bottom"})
                    for s in sorted_slices_attr[-n_select:]:
                        selected_slices.append({**s, "selection": "attr_top"})

                except Exception as e:
                    print(f"Attribution failed: {e}")
                    # Fallback if attribution fails
                    for s in sorted_slices_prob[:n_select]:
                        selected_slices.append({**s, "selection": "prob_bottom"})
                    for s in sorted_slices_prob[-n_select:]:
                        selected_slices.append({**s, "selection": "prob_top"})

        fold_pbar.set_postfix({"slices": len(selected_slices)})

    macro_f1 = f1_score(all_slice_true, all_slice_pred, average="macro")
    per_class_f1 = f1_score(all_slice_true, all_slice_pred, average=None)
    per_class_precision = precision_score(
        all_slice_true, all_slice_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        all_slice_true, all_slice_pred, average=None, zero_division=0
    )

    whole_macro_f1 = f1_score(all_whole_true, all_whole_pred, average="macro")
    whole_per_class_f1 = f1_score(all_whole_true, all_whole_pred, average=None)
    whole_per_class_precision = precision_score(
        all_whole_true, all_whole_pred, average=None, zero_division=0
    )
    whole_per_class_recall = recall_score(
        all_whole_true, all_whole_pred, average=None, zero_division=0
    )

    def _baseline_metrics(true_labels: list, strategy: str) -> tuple:
        """Return (macro_f1, per_class_f1, per_class_precision, per_class_recall) for a baseline."""
        if strategy == "random":
            return stratified_random_baseline(true_labels)
        classes = sorted(set(true_labels))
        majority = max(classes, key=true_labels.count)
        pred = [majority] * len(true_labels)
        return (
            f1_score(true_labels, pred, average="macro"),
            f1_score(true_labels, pred, average=None),
            precision_score(true_labels, pred, average=None, zero_division=0),
            recall_score(true_labels, pred, average=None, zero_division=0),
        )

    def _print_metrics(label: str, mf1, pf1, pprec, prec_):
        print(f"{label} Macro F1: {mf1:.4f}")
        for c in range(len(pf1)):
            print(
                f"  - Class {c}: F1={pf1[c]:.4f}, Precision={pprec[c]:.4f}, Recall={prec_[c]:.4f}"
            )

    print(f"\nFinished LOO CV. Total selected slices collected: {len(selected_slices)}")

    print(f"\n--- Per-Slice Metrics ---")
    _print_metrics(
        "Model", macro_f1, per_class_f1, per_class_precision, per_class_recall
    )
    _print_metrics("Majority baseline", *_baseline_metrics(all_slice_true, "majority"))
    _print_metrics(
        "Random (stratified) baseline", *_baseline_metrics(all_slice_true, "random")
    )

    print(f"\n--- Whole-Input Metrics ---")
    _print_metrics(
        "Model",
        whole_macro_f1,
        whole_per_class_f1,
        whole_per_class_precision,
        whole_per_class_recall,
    )
    _print_metrics("Majority baseline", *_baseline_metrics(all_whole_true, "majority"))
    _print_metrics(
        "Random (stratified) baseline", *_baseline_metrics(all_whole_true, "random")
    )

    # Save or inspect the resulting slices to see patterns indicating the two classes
    if selected_slices:
        out_file = f"correct_slices_{args.data_age}_{args.target_class}.pt"
        torch.save(selected_slices, out_file)
        print(f"Saved selected slice feature arrays to {out_file}")


if __name__ == "__main__":
    main()
