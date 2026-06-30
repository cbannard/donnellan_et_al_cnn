import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze locally saved correct slices from sliding window evaluation"
    )
    parser.add_argument(
        "--slices-file", type=Path, default=Path("correct_slices_11m_top.pt")
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        help="A representative time series file to get channel names",
        default=Path(
            "Donnellan_et_al_Data/Donnellan_et_al_Data/time_series_data/p13_11m.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("slice_analysis"))
    args = parser.parse_args()

    if not args.slices_file.exists():
        print(
            f"Error: Could not find '{args.slices_file}'. Did you run the training script first?"
        )
        return

    # Load channel names
    channels = None
    if args.channels_file.exists():
        frame = pd.read_csv(args.channels_file, index_col=0)
        frame.index = frame.index.astype(str).str.strip().str.strip('"')
        channels = [lbl for lbl in frame.index if lbl not in {"OFFSHOT", "PADDING"}]
        print(f"Loaded {len(channels)} channels from reference file.")

    slices_data = torch.load(args.slices_file, weights_only=False)
    print(f"Loaded {len(slices_data)} slices.")

    if not slices_data:
        print("No slices to analyze.")
        return

    # Group by class dynamically
    unique_classes = sorted(list(set(s["class"] for s in slices_data)))

    # Dictionary to keep the top 30 filtered slice lists to reuse below
    filtered_class_slices = {}

    for c in unique_classes:
        # Get all slices for the class
        c_slices = [s for s in slices_data if s["class"] == c]

        if c_slices and "selection" in c_slices[0]:
            prob_top = [
                s["slice_features"]
                for s in c_slices
                if s.get("selection") == "prob_top"
            ]
            prob_bottom = [
                s["slice_features"]
                for s in c_slices
                if s.get("selection") == "prob_bottom"
            ]
            attr_top = [
                s["slice_features"]
                for s in c_slices
                if s.get("selection") == "attr_top"
            ]
            attr_bottom = [
                s["slice_features"]
                for s in c_slices
                if s.get("selection") == "attr_bottom"
            ]

            if prob_top:
                filtered_class_slices[f"{c}_prob_top"] = prob_top
            if prob_bottom:
                filtered_class_slices[f"{c}_prob_bottom"] = prob_bottom
            if attr_top:
                filtered_class_slices[f"{c}_attr_top"] = attr_top
            if attr_bottom:
                filtered_class_slices[f"{c}_attr_bottom"] = attr_bottom

            print(f"Class {c}: {len(c_slices)} total slices.")
            print(f"  -> Prob Top: {len(prob_top)}, Prob Bottom: {len(prob_bottom)}")
            print(f"  -> Attr Top: {len(attr_top)}, Attr Bottom: {len(attr_bottom)}")
        elif c_slices and "prob_class1" in c_slices[0]:
            top_all = []
            bottom_all = []
            participants = set(s["participant"] for s in c_slices)

            for p in participants:
                p_slices = [s for s in c_slices if s["participant"] == p]
                p_slices = sorted(
                    p_slices, key=lambda x: x["prob_class1"], reverse=True
                )
                top_all.extend(p_slices[:20])
                bottom_all.extend(p_slices[-20:])

            filtered_class_slices[f"{c}_prob_top"] = [
                s["slice_features"] for s in top_all
            ]
            filtered_class_slices[f"{c}_prob_bottom"] = [
                s["slice_features"] for s in bottom_all
            ]
            print(f"Class {c}: {len(c_slices)} slices.")
            print(
                f"  -> Extracted top 20 and bottom 20 class-1 probabilities per participant "
                f"(Top: {len(top_all)}, Bottom: {len(bottom_all)})."
            )
        else:
            filtered_class_slices[str(c)] = [s["slice_features"] for s in c_slices]

    args.out_dir.mkdir(exist_ok=True)

    def process_class(slice_list, class_label):
        if not slice_list:
            return

        stack = np.stack(slice_list)  # (N, Channels, Time)

        # Calculate mean and standard error
        mean = np.mean(stack, axis=0)  # (Channels, Time)
        std = np.std(stack, axis=0)
        se = std / np.sqrt(stack.shape[0])

        # Determine channels if not loaded
        num_channels = stack.shape[1]
        chan_names = (
            channels
            if channels and len(channels) == num_channels
            else [f"C{i}" for i in range(num_channels)]
        )

        # Save means
        mean_df = pd.DataFrame(mean, index=chan_names)
        mean_df.to_csv(args.out_dir / f"class_{class_label}_mean.csv")

        # Save standard errors
        se_df = pd.DataFrame(se, index=chan_names)
        se_df.to_csv(args.out_dir / f"class_{class_label}_se.csv")

        # Create plot
        plt.figure(figsize=(10, 6))
        time_steps = np.arange(mean.shape[1])

        for i, chan in enumerate(chan_names):
            plt.plot(time_steps, mean[i], label=chan)
            plt.fill_between(time_steps, mean[i] - se[i], mean[i] + se[i], alpha=0.2)

        plt.title(f"Class {class_label} - Mean Patterns with Standard Error")
        plt.xlabel("Time Step (within window)")
        plt.ylabel("Standardized Channel Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(args.out_dir / f"class_{class_label}_plot.png", dpi=300)
        plt.close()

        # Create heatmap plot
        plt.figure(figsize=(10, 6))
        plt.imshow(mean, aspect="auto", cmap="RdBu_r", origin="lower")
        plt.colorbar(label="Mean Standardized Value")
        plt.yticks(ticks=np.arange(len(chan_names)), labels=chan_names)
        plt.title(f"Class {class_label} - Receptive Field Mean Heatmap")
        plt.xlabel("Time Step (within window)")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.savefig(args.out_dir / f"class_{class_label}_heatmap.png", dpi=300)
        plt.close()

        return mean, chan_names

    # Collect computed means for subplots
    computed_means = {}
    chan_names_ref = None

    for label_key, slices_list in filtered_class_slices.items():
        res = process_class(slices_list, label_key)
        if res:
            computed_means[label_key] = res[0]
            if chan_names_ref is None:
                chan_names_ref = res[1]

    # Combined Heatmap plotting for each class
    for c in unique_classes:
        for prefix in ["prob", "attr"]:
            top_key = f"{c}_{prefix}_top"
            bottom_key = f"{c}_{prefix}_bottom"
            if top_key in computed_means and bottom_key in computed_means:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                vmin = min(
                    computed_means[top_key].min(), computed_means[bottom_key].min()
                )
                vmax = max(
                    computed_means[top_key].max(), computed_means[bottom_key].max()
                )

                im1 = axes[0].imshow(
                    computed_means[top_key],
                    aspect="auto",
                    cmap="RdBu_r",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[0].set_yticks(
                    ticks=np.arange(len(chan_names_ref)), labels=chan_names_ref
                )

                title_metric = (
                    "Class-1 Probability" if prefix == "prob" else "Total Attribution"
                )
                axes[0].set_title(f"Class {c} - High {title_metric}")
                axes[0].set_xlabel("Time Step (within window)")
                axes[0].set_ylabel("Channel")

                im2 = axes[1].imshow(
                    computed_means[bottom_key],
                    aspect="auto",
                    cmap="RdBu_r",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[1].set_yticks(ticks=np.arange(len(chan_names_ref)), labels=[])
                axes[1].set_title(f"Class {c} - Low {title_metric}")
                axes[1].set_xlabel("Time Step (within window)")

                fig.colorbar(
                    im2, ax=axes.ravel().tolist(), label="Mean Standardized Value"
                )
                plt.savefig(
                    args.out_dir / f"class_{c}_{prefix}_combined_heatmap.png", dpi=300
                )
                plt.close()

    print(f"Saved mean, standard error matrices and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
