import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CLASS_NAMES = [
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
]

CLASS_NAME_TO_ID = {class_name: index + 1 for index, class_name in enumerate(CLASS_NAMES)}


@dataclass
class PolygonAnnotation:
    """Store a single polygon annotation.

    Args:
        label: Class label from the LabelMe shape.
        points: Polygon points in absolute pixel coordinates.
    """

    label: str
    points: List[Tuple[float, float]]


@dataclass
class LabelMeSample:
    """Store one LabelMe image and its polygon annotations.

    Args:
        image_path: Absolute path to the paired image file.
        json_path: Absolute path to the LabelMe JSON file.
        image_name: File name used when writing converted datasets.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        annotations: Polygon annotations for this image.
        primary_label: Label used for stratified splitting.
    """

    image_path: Path
    json_path: Path
    image_name: str
    image_width: int
    image_height: int
    annotations: List[PolygonAnnotation]
    primary_label: str


def load_labelme_samples(raw_dir: Path) -> List[LabelMeSample]:
    """Load LabelMe JSON files and paired images.

    Args:
        raw_dir: Directory containing LabelMe JSON files and paired images.

    Returns:
        Loaded samples sorted by JSON file name.
    """

    samples = []
    for json_path in sorted(raw_dir.glob("*.json")):
        with json_path.open("r", encoding = "utf-8") as file_obj:
            labelme_data = json.load(file_obj)

        annotations = []
        for shape in labelme_data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue

            label = shape.get("label", "")
            if label not in CLASS_NAME_TO_ID:
                raise ValueError(f"Unknown class label {label!r} in {json_path}")

            points = [(float(x), float(y)) for x, y in shape.get("points", [])]
            if len(points) < 3:
                continue

            annotations.append(PolygonAnnotation(label = label, points = points))

        if not annotations:
            raise ValueError(f"No valid polygon annotations found in {json_path}")

        image_name = labelme_data.get("imagePath") or f"{json_path.stem}.PNG"
        image_path = raw_dir / image_name
        if not image_path.exists():
            image_path = _find_image_path(raw_dir = raw_dir, stem = json_path.stem)

        samples.append(
            LabelMeSample(
                image_path = image_path,
                json_path = json_path,
                image_name = image_path.name,
                image_width = int(labelme_data["imageWidth"]),
                image_height = int(labelme_data["imageHeight"]),
                annotations = annotations,
                primary_label = annotations[0].label,
            )
        )

    return samples


def split_samples_by_class(
    samples: List[LabelMeSample],
    seed: int,
    train_per_class: int = 9,
    valid_per_class: int = 1,
    test_per_class: int = 0,
) -> Dict[str, List[LabelMeSample]]:
    """Split samples by primary class into train, valid, and test groups.

    Args:
        samples: Loaded LabelMe samples.
        seed: Random seed for deterministic shuffling.
        train_per_class: Number of train images to keep for each class.
        valid_per_class: Number of valid images to keep for each class.
        test_per_class: Number of test images to keep for each class.

    Returns:
        Mapping with split names and samples.
    """

    random_generator = random.Random(seed)
    samples_by_class = {class_name: [] for class_name in CLASS_NAMES}
    for sample in samples:
        samples_by_class[sample.primary_label].append(sample)

    split_counts = {
        "train": train_per_class,
        "valid": valid_per_class,
        "test": test_per_class,
    }
    splits = {split_name: [] for split_name, count in split_counts.items() if count > 0}
    for class_name in CLASS_NAMES:
        class_samples = sorted(samples_by_class[class_name], key = lambda sample: sample.image_name)
        expected_count = sum(split_counts.values())
        if len(class_samples) < expected_count:
            raise ValueError(
                f"{class_name} has {len(class_samples)} samples, "
                f"but {expected_count} are required"
            )

        random_generator.shuffle(class_samples)
        start_index = 0
        for split_name, split_count in split_counts.items():
            if split_count <= 0:
                continue

            end_index = start_index + split_count
            splits[split_name].extend(class_samples[start_index:end_index])
            start_index = end_index

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key = lambda sample: sample.image_name)

    return splits


def summarize_splits(splits: Dict[str, List[LabelMeSample]]) -> Dict[str, Dict[str, object]]:
    """Summarize image, annotation, and class counts by split.

    Args:
        splits: Mapping from split name to samples.

    Returns:
        Summary dictionary keyed by split name.
    """

    summary = {}
    for split_name, samples in splits.items():
        class_counts = {class_name: 0 for class_name in CLASS_NAMES}
        annotation_count = 0
        for sample in samples:
            for annotation in sample.annotations:
                class_counts[annotation.label] += 1
                annotation_count += 1

        summary[split_name] = {
            "images": len(samples),
            "annotations": annotation_count,
            "class_counts": class_counts,
        }

    return summary


def _find_image_path(raw_dir: Path, stem: str) -> Path:
    """Find a paired image path by file stem.

    Args:
        raw_dir: Directory containing source images.
        stem: Image file stem to search for.

    Returns:
        Matching image path.
    """

    supported_suffixes = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]
    for suffix in supported_suffixes:
        image_path = raw_dir / f"{stem}{suffix}"
        if image_path.exists():
            return image_path

    raise FileNotFoundError(f"Cannot find paired image for {stem} in {raw_dir}")
