import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.append(os.getcwd())

from rfdetr import RFDETRSegNano


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description = "Train RF-DETR Seg Nano.")
    parser.add_argument(
        "--dataset-dir",
        type = Path,
        default = Path("data/train_data_seg_6classes_60case_test_v0/train_coco"),
    )
    parser.add_argument("--output-dir", type = Path, default = Path("runs/rfdetr_seg_nano"))
    parser.add_argument("--rf-home", type = Path, default = Path("weights/rfdetr"))
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--grad-accum-steps", type = int, default = 4)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--device", type = str, default = "cpu")
    parser.add_argument("--seed", type = int, default = 42)

    return parser.parse_args()


def main() -> None:
    """Run RF-DETR Seg Nano training.

    Args:
        None.
    """

    args = parse_args()
    rf_home = args.rf_home.resolve()
    os.environ["RF_HOME"] = str(rf_home)
    rf_home.mkdir(parents = True, exist_ok = True)
    args.output_dir.mkdir(parents = True, exist_ok = True)

    logger.info("=" * 80)
    logger.info("Starting RF-DETR Seg Nano training")
    logger.info("=" * 80)
    logger.info("Dataset directory: %s", args.dataset_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("RF_HOME: %s", rf_home)
    logger.info(
        "Training params: epochs=%s batch_size=%s grad_accum_steps=%s lr=%s device=%s seed=%s",
        args.epochs,
        args.batch_size,
        args.grad_accum_steps,
        args.lr,
        args.device,
        args.seed,
    )

    model = RFDETRSegNano()
    model.maybe_download_pretrain_weights()
    # model.train(
    #     dataset_dir = str(args.dataset_dir),
    #     epochs = args.epochs,
    #     batch_size = args.batch_size,
    #     grad_accum_steps = args.grad_accum_steps,
    #     lr = args.lr,
    #     output_dir = str(args.output_dir),
    #     device = args.device,
    #     seed = args.seed,
    #     resolution = 432
    # )
    print("resolution:", model.model_config.resolution)
    print("patch_size:", model.model_config.patch_size)
    print("num_windows:", model.model_config.num_windows)

    block_size = model.model_config.patch_size * model.model_config.num_windows
    print("block_size:", block_size)

    candidate_resolutions = [312, 384, 432, 504, 624, 768]
    valid_resolutions = [
        resolution
        for resolution in candidate_resolutions
        if resolution % block_size == 0
    ]

    print("valid_resolutions:", valid_resolutions)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )
    main()
