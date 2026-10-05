import argparse
import os
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the CAD dataset layout expected by eval_MoCA-Mask.py."
    )
    parser.add_argument(
        "--source",
        default="/root/autodl-tmp/A_large/CamouflagedAnimalDataset_before",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/A_large/CamouflagedAnimalDataset_eval",
    )
    return parser.parse_args()


def image_stems(folder):
    return {
        path.stem
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def replace_symlink(link, target):
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace existing non-symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    frame_root = source / "filtered_frames"
    gt_root = source / "new_gt"
    output = Path(args.output).resolve()

    frame_videos = {path.name for path in frame_root.iterdir() if path.is_dir()}
    gt_videos = {path.name for path in gt_root.iterdir() if path.is_dir()}
    if frame_videos != gt_videos:
        raise ValueError(
            f"Video mismatch: frames-only={sorted(frame_videos - gt_videos)}, "
            f"GT-only={sorted(gt_videos - frame_videos)}"
        )

    output.mkdir(parents=True, exist_ok=True)
    for video in sorted(frame_videos):
        frames = frame_root / video / "frames"
        groundtruth = gt_root / video / "groundtruth"
        if not frames.is_dir() or not groundtruth.is_dir():
            raise FileNotFoundError(f"Missing frames or groundtruth for {video}")

        frame_ids = image_stems(frames)
        gt_ids = image_stems(groundtruth)
        if frame_ids != gt_ids:
            raise ValueError(
                f"{video}: frame/GT mismatch; "
                f"frames-only={sorted(frame_ids - gt_ids)[:10]}, "
                f"GT-only={sorted(gt_ids - frame_ids)[:10]}"
            )

        video_output = output / video
        video_output.mkdir(exist_ok=True)
        replace_symlink(video_output / "Imgs", frames)
        replace_symlink(video_output / "GT", groundtruth)
        print(f"{video}: {len(frame_ids)} matched frames")

    print(f"Prepared {len(frame_videos)} CAD videos at {output}")
    print("Original images and masks were not copied or modified.")


if __name__ == "__main__":
    main()
