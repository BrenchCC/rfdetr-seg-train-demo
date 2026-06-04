import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.data_loader import CLASS_NAMES, CLASS_NAME_TO_ID, LabelMeSample


def convert_splits_to_coco(
    splits: Dict[str, List[LabelMeSample]],
    output_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """Write LabelMe samples as a COCO segmentation dataset.

    Args:
        splits: Mapping from split name to loaded samples.
        output_dir: Destination directory for the COCO dataset.

    Returns:
        Conversion statistics keyed by split name.
    """

    output_dir.mkdir(parents = True, exist_ok = True)
    stats = {}
    for split_name, samples in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents = True, exist_ok = True)

        coco_data = _build_coco_data(samples = samples, split_dir = split_dir)
        annotation_path = split_dir / "_annotations.coco.json"
        with annotation_path.open("w", encoding = "utf-8") as file_obj:
            json.dump(coco_data, file_obj, indent = 2)

        stats[split_name] = {
            "images": len(coco_data["images"]),
            "annotations": len(coco_data["annotations"]),
        }

    return stats


def _build_coco_data(samples: List[LabelMeSample], split_dir: Path) -> Dict[str, object]:
    """Build one COCO annotation dictionary and copy images.

    Args:
        samples: Samples for one split.
        split_dir: Destination split directory.

    Returns:
        COCO annotation dictionary.
    """

    coco_data = {
        "info": {
            "description": "RF-DETR segmentation dataset converted from LabelMe",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {"id": CLASS_NAME_TO_ID[class_name], "name": class_name, "supercategory": "object"}
            for class_name in CLASS_NAMES
        ],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    for image_id, sample in enumerate(
        tqdm(samples, desc = f"COCO {split_dir.name}", unit = "image"),
        start = 1,
    ):
        destination_image_path = split_dir / sample.image_name
        shutil.copy2(sample.image_path, destination_image_path)

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": sample.image_name,
                "width": sample.image_width,
                "height": sample.image_height,
            }
        )

        for annotation in sample.annotations:
            polygon = _flatten_points(points = annotation.points)
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CLASS_NAME_TO_ID[annotation.label],
                    "segmentation": [polygon],
                    "area": _polygon_area(points = annotation.points),
                    "bbox": _polygon_bbox(points = annotation.points),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco_data


def _flatten_points(points: List[Tuple[float, float]]) -> List[float]:
    """Flatten polygon points for COCO segmentation.

    Args:
        points: Polygon points in absolute pixel coordinates.

    Returns:
        Flat list of x and y values.
    """

    flattened_points = []
    for x_value, y_value in points:
        flattened_points.extend([x_value, y_value])

    return flattened_points


def _polygon_bbox(points: List[Tuple[float, float]]) -> List[float]:
    """Calculate COCO bbox from polygon points.

    Args:
        points: Polygon points in absolute pixel coordinates.

    Returns:
        Bounding box as x, y, width, and height.
    """

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_values)
    y_max = max(y_values)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    """Calculate polygon area with the shoelace formula.

    Args:
        points: Polygon points in absolute pixel coordinates.

    Returns:
        Absolute polygon area.
    """

    area = 0.0
    point_count = len(points)
    for index in range(point_count):
        x_current, y_current = points[index]
        x_next, y_next = points[(index + 1) % point_count]
        area += x_current * y_next - x_next * y_current

    return abs(area) / 2.0
