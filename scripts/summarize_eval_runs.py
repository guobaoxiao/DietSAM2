import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "S-measure",
    "Weighted F-measure",
    "MAE",
    "F-measure",
    "E-measure",
    "Dice",
    "IoU",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize multiple DietSAM2 evaluation runs as mean +/- std."
    )
    parser.add_argument("result_csv", nargs="+", help="Evaluation result.csv files")
    parser.add_argument("--output", default="cad_full_mean_std.csv")
    return parser.parse_args()


def load_result(path):
    frame = pd.read_csv(path, sep="\t", index_col=0)
    missing = [column for column in ["prompt_type", *METRICS] if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame.set_index("prompt_type")[METRICS].sort_index()


def main():
    args = parse_args()
    paths = [Path(path) for path in args.result_csv]
    frames = [load_result(path) for path in paths]

    reference_index = frames[0].index
    if any(not frame.index.equals(reference_index) for frame in frames[1:]):
        raise ValueError("Prompt types differ across result files")

    values = pd.concat(frames, keys=range(len(frames)), names=["run", "prompt_type"])
    grouped = values.groupby(level="prompt_type", sort=False)
    mean = grouped.mean() * 100
    std = grouped.std(ddof=0) * 100

    summary = pd.DataFrame(index=reference_index)
    for metric in METRICS:
        summary[metric] = [
            f"{mean.loc[prompt, metric]:.1f} +/- {std.loc[prompt, metric]:.1f}"
            for prompt in reference_index
        ]

    summary.index.name = "prompt_type"
    summary.to_csv(args.output)
    print(summary.to_string())
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
