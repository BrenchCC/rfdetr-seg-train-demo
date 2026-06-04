import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.data_loader import CLASS_NAMES, CLASS_NAME_TO_ID, LabelMeSample


def convert_splits_to_yolo(
    splits: Dict[str, List[LabelMeSample]],
    output_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """Write LabelMe samples as a YOLO segmentation dataset.

    Args:
        splits: Mapping from split name to loaded samples.
        output_dir: Destination directory for the YOLO dataset.

    Returns:
        Conversion statistics keyed by split name.
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents = True, exist_ok = True)
    _write_data_yaml(
        output_dir = output_dir,
        has_valid = "valid" in splits,
        has_test = "test" in splits,
    )

    stats = {}
    for split_name, samples in splits.items():
        image_dir = output_dir / split_name / "images"
        label_dir = output_dir / split_name / "labels"
        image_dir.mkdir(parents = True, exist_ok = True)
        label_dir.mkdir(parents = True, exist_ok = True)

        annotation_count = 0
        for sample in tqdm(samples, desc = f"YOLO {split_name}", unit = "image"):
            shutil.copy2(sample.image_path, image_dir / sample.image_name)
            yolo_lines = []
            for annotation in sample.annotations:
                yolo_lines.append(
                    _annotation_to_yolo_line(
                        label = annotation.label,
                        points = annotation.points,
                        image_width = sample.image_width,
                        image_height = sample.image_height,
                    )
                )
                annotation_count += 1

            label_path = label_dir / f"{Path(sample.image_name).stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding = "utf-8")

        stats[split_name] = {"images": len(samples), "annotations": annotation_count}

    return stats


def _write_data_yaml(output_dir: Path, has_valid: bool, has_test: bool) -> None:
    """Write YOLO data.yaml.

    Args:
        output_dir: Destination directory for the YOLO dataset.
        has_valid: Whether to include a valid image path.
        has_test: Whether to include a test image path.
    """

    valid_text = "val: valid/images\n" if has_valid else ""
    test_text = "test: test/images\n" if has_test else ""
    names_text = "\n".join(
        f"  {CLASS_NAME_TO_ID[class_name] - 1}: {class_name}" for class_name in CLASS_NAMES
    )
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: train/images\n"
        f"{valid_text}"
        f"{test_text}"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names_text}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding = "utf-8")


def _annotation_to_yolo_line(
    label: str,
    points: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
) -> str:
    """Convert one polygon annotation to a YOLO segmentation line.

    Args:
        label: Class label from the LabelMe shape.
        points: Polygon points in absolute pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        YOLO segmentation line with normalized polygon coordinates.
    """

    class_id = CLASS_NAME_TO_ID[label] - 1
    normalized_points = []
    for x_value, y_value in points:
        normalized_points.extend(
            [
                _normalize_coordinate(value = x_value, size = image_width),
                _normalize_coordinate(value = y_value, size = image_height),
            ]
        )

    coordinates = " ".join(f"{value:.6f}" for value in normalized_points)
    return f"{class_id} {coordinates}"


def _normalize_coordinate(value: float, size: int) -> float:
    """Normalize and clamp one coordinate.

    Args:
        value: Absolute pixel coordinate.
        size: Image width or height in pixels.

    Returns:
        Coordinate normalized to the range from 0 to 1.
    """

    return min(max(value / float(size), 0.0), 1.0)
