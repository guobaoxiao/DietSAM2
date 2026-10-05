import argparse
import json
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from new_eval import calculate_metrics, convert_ndarray_to_list, parse_result_json
from sam2.build_sam import build_dietsam2_video_predictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROMPT_ALIASES = {"click": "point", "bbox": "box"}


def numeric_sort_key(filename):
    match = re.search(r"\d+", filename)
    if match is None:
        raise ValueError(f"Filename must contain a frame number: {filename}")
    return int(match.group())


def list_images(folder, extensions):
    return sorted(
        (
            name
            for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in extensions
        ),
        key=numeric_sort_key,
    )


def load_video_data(test_data_folder, video_name):
    video_dir = os.path.join(test_data_folder, video_name, "Imgs")
    gt_dir = os.path.join(test_data_folder, video_name, "GT")
    if not os.path.isdir(video_dir) or not os.path.isdir(gt_dir):
        raise FileNotFoundError(
            f"{video_name} must contain both Imgs/ and GT/ directories"
        )

    frame_names = list_images(video_dir, FRAME_EXTENSIONS)
    gt_names = list_images(gt_dir, IMAGE_EXTENSIONS)
    if len(frame_names) != len(gt_names):
        raise ValueError(
            f"{video_name}: {len(frame_names)} frames but {len(gt_names)} masks"
        )

    ground_truth = np.stack(
        [
            np.asarray(Image.open(os.path.join(gt_dir, name)).convert("L"))
            for name in gt_names
        ]
    )
    return video_dir, gt_names, ground_truth


def build_prompt(prompt_type, first_mask):
    if prompt_type == "mask":
        return {"mask": first_mask}

    foreground = np.argwhere(first_mask > 0)
    if foreground.size == 0:
        raise ValueError("The first-frame ground-truth mask is empty")

    if prompt_type == "box":
        y_indices, x_indices = foreground[:, 0], foreground[:, 1]
        return {
            "box": np.array(
                [
                    x_indices.min(),
                    y_indices.min(),
                    x_indices.max(),
                    y_indices.max(),
                ],
                dtype=np.float32,
            )
        }

    random.seed(42)
    y, x = random.choice(foreground)
    return {
        "points": np.array([[x, y]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int32),
    }


def propagate_video(predictor, video_dir, output_mode, prompt_type, first_mask):
    inference_state = predictor.init_state(
        video_path=video_dir,
        output_mode=output_mode,
    )
    predictor.reset_state(inference_state)

    common_args = {
        "inference_state": inference_state,
        "frame_idx": 0,
        "obj_id": 1,
    }
    prompt = build_prompt(prompt_type, first_mask)
    if prompt_type == "mask":
        predictor.add_new_mask(**common_args, **prompt)
    else:
        predictor.add_new_points_or_box(**common_args, **prompt)

    predictions = [
        (mask_logits[0, 0] > 0).cpu().numpy()
        for _, _, mask_logits in predictor.propagate_in_video(inference_state)
    ]
    return np.asarray(predictions)


def save_predictions(
    predictions,
    gt_names,
    output_path,
    video_name,
    prompt_type,
):
    prompt_folder = {
        "point": "click_prompt",
        "box": "box_prompt",
        "mask": "mask_prompt",
    }[prompt_type]
    save_dir = os.path.join(output_path, prompt_folder, video_name)
    os.makedirs(save_dir, exist_ok=True)
    for filename, prediction in zip(gt_names, predictions):
        cv2.imwrite(
            os.path.join(save_dir, filename),
            prediction.astype(np.uint8) * 255,
        )


def evaluate_video(
    video_name,
    predictor,
    output_mode,
    prompt_type,
    test_data_folder,
    output_path,
):
    video_dir, gt_names, ground_truth = load_video_data(
        test_data_folder,
        video_name,
    )
    predictions = propagate_video(
        predictor,
        video_dir,
        output_mode,
        prompt_type,
        ground_truth[0],
    )
    if len(predictions) != len(ground_truth):
        raise RuntimeError(
            f"{video_name}: model returned {len(predictions)} masks for "
            f"{len(ground_truth)} frames"
        )
    save_predictions(
        predictions,
        gt_names,
        output_path,
        video_name,
        prompt_type,
    )
    return calculate_metrics(predictions, ground_truth)


def parse_prompt_types(raw_prompt_types):
    prompt_types = []
    for value in raw_prompt_types.split(","):
        prompt_type = value.strip().lower()
        prompt_type = PROMPT_ALIASES.get(prompt_type, prompt_type)
        if prompt_type not in {"mask", "box", "point"}:
            raise ValueError(
                f"Unsupported prompt type '{prompt_type}'. "
                "Choose from mask, box, and point."
            )
        if prompt_type not in prompt_types:
            prompt_types.append(prompt_type)
    return prompt_types


def parse_args():
    parser = argparse.ArgumentParser("DietSAM2 Evaluation")
    parser.add_argument("--model_cfg", default="sam2_hiera_t.yaml")
    parser.add_argument(
        "--ckpt_path",
        default="work_dir_2605/0521test/DietSAM2_epoch_16_merged.pth",
    )
    parser.add_argument(
        "--output_mode",
        default="combined_mask",
        choices=["original_sam2_mask", "combined_mask"],
    )
    parser.add_argument(
        "--data_path",
        default="/root/autodl-tmp/A_large/MoCA_Video/TestDataset_per_sq",
    )
    parser.add_argument("--prompt_types", default="mask,box,point")
    parser.add_argument("--output_path", default="eval_result/0521test/")
    parser.add_argument("--dietsam2_extra", "--camsam2_extra", dest="dietsam2_extra")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS support is preliminary; results may differ from CUDA.")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def list_videos(data_path):
    return sorted(
        name
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    )


def save_summary(output_path, prompt_types):
    rows = []
    for prompt_type in prompt_types:
        result_path = os.path.join(
            output_path,
            f"MoCA-Mask_{prompt_type}.json",
        )
        row = parse_result_json(result_path)
        row["prompt_type"] = prompt_type
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results[
        ["prompt_type"] + [column for column in results if column != "prompt_type"]
    ]
    results = results.drop(
        columns=["BIoU", "TIoU", "Boundary Accuracy"],
        errors="ignore",
    ).round(3)
    results.to_csv(
        os.path.join(output_path, "result.csv"),
        sep="\t",
        encoding="utf-8",
    )
    print(results)


def main():
    args = parse_args()
    prompt_types = parse_prompt_types(args.prompt_types)
    videos = list_videos(args.data_path)
    if not videos:
        raise FileNotFoundError(f"No video directories found in {args.data_path}")

    os.makedirs(args.output_path, exist_ok=True)
    predictor_args = {
        "device": get_device(),
    }
    if args.dietsam2_extra is not None:
        predictor_args["dietsam2_extra"] = args.dietsam2_extra
    predictor = build_dietsam2_video_predictor(
        args.model_cfg,
        args.ckpt_path,
        **predictor_args,
    )

    for prompt_type in prompt_types:
        results = {}
        for video_name in videos:
            metrics = evaluate_video(
                video_name,
                predictor,
                args.output_mode,
                prompt_type,
                args.data_path,
                args.output_path,
            )
            results[video_name] = convert_ndarray_to_list(metrics)

        result_path = os.path.join(
            args.output_path,
            f"MoCA-Mask_{prompt_type}.json",
        )
        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

    save_summary(args.output_path, prompt_types)


if __name__ == "__main__":
    main()
