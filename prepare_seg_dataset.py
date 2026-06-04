import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.append(os.getcwd())

from utils.convert_coco import convert_splits_to_coco
from utils.convert_yolo import convert_splits_to_yolo
from utils.data_loader import load_labelme_samples, split_samples_by_class, summarize_splits


logger = logging.getLogger(__name__)


DEFAULT_DATA_ROOT = Path("data/train_data_seg_6classes_60case_test_v0")
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description = "Prepare COCO and YOLO segmentation datasets.")
    parser.add_argument("--raw-dir", type = Path, default = DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type = Path, default = DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--train-per-class", type = int, default = 9)
    parser.add_argument("--valid-per-class", type = int, default = 1)
    parser.add_argument("--test-per-class", type = int, default = 0)
    parser.add_argument(
        "--formats",
        nargs = "+",
        choices = ["coco", "yolo"],
        default = ["coco", "yolo"],
    )

    return parser.parse_args()


def main() -> None:
    """Prepare segmentation datasets.

    Args:
        None.
    """

    args = parse_args()
    _validate_args(args = args)
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    coco_output_dir = output_dir / "train_coco"
    yolo_output_dir = output_dir / "train_yolo"

    logger.info("=" * 80)
    logger.info("Preparing RF-DETR segmentation datasets")
    logger.info("=" * 80)
    logger.info("Raw directory: %s", raw_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Formats: %s", ", ".join(args.formats))
    logger.info(
        "Split per class: train=%s valid=%s test=%s",
        args.train_per_class,
        args.valid_per_class,
        args.test_per_class,
    )

    samples = load_labelme_samples(raw_dir = raw_dir)
    splits = split_samples_by_class(
        samples = samples,
        seed = args.seed,
        train_per_class = args.train_per_class,
        valid_per_class = args.valid_per_class,
        test_per_class = args.test_per_class,
    )
    split_summary = summarize_splits(splits = splits)
    _log_split_summary(split_summary = split_summary)

    if "coco" in args.formats:
        logger.info("-" * 60)
        logger.info("Writing COCO segmentation dataset")
        logger.info("-" * 60)
        coco_stats = convert_splits_to_coco(splits = splits, output_dir = coco_output_dir)
        logger.info("COCO output: %s", coco_output_dir)
        logger.info("COCO stats: %s", coco_stats)

    if "yolo" in args.formats:
        logger.info("-" * 60)
        logger.info("Writing YOLO segmentation dataset")
        logger.info("-" * 60)
        yolo_stats = convert_splits_to_yolo(splits = splits, output_dir = yolo_output_dir)
        logger.info("YOLO output: %s", yolo_output_dir)
        logger.info("YOLO stats: %s", yolo_stats)

    logger.info("=" * 80)
    logger.info("Dataset preparation finished")
    logger.info("=" * 80)


def _log_split_summary(split_summary: dict) -> None:
    """Log split summary.

    Args:
        split_summary: Summary dictionary returned by summarize_splits.
    """

    logger.info("-" * 60)
    logger.info("Split summary")
    logger.info("-" * 60)
    for split_name, stats in split_summary.items():
        logger.info(
            "%s: images=%s annotations=%s class_counts=%s",
            split_name,
            stats["images"],
            stats["annotations"],
            stats["class_counts"],
        )


def _validate_args(args: argparse.Namespace) -> None:
    """Validate split arguments.

    Args:
        args: Parsed command line arguments.
    """

    split_counts = [args.train_per_class, args.valid_per_class, args.test_per_class]
    if any(split_count < 0 for split_count in split_counts):
        raise ValueError("Split counts must be non-negative")

    if args.train_per_class == 0:
        raise ValueError("train_per_class must be greater than 0")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )
    main()
