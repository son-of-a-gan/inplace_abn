import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse

from segmentation_module import SegmentationModule, load_snapshot, flip


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Testing script for the Vistas segmentation model")
    parser.add_argument("--snapshot", type=str, default="weights/wide_resnet38_deeplab_vistas.pth",
                        help="Snapshot file to load")
    parser.add_argument("--in-gt", type=str, default="folder_gt",
                        help="Path to folder containing sharp ground truth images.")
    parser.add_argument("--in-fake", type=str, default="folder_fake",
                        help="Path to folder containing deblurred images.")
    parser.add_argument("--out-dir", type=str, default="folder_out",
                        help="Path to output folder.")
    # minimally needed
    parser.add_argument("--rank", metavar="RANK",
                        type=int, default=0, help="GPU id")
    parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                        help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
    # not sure uses below here
    parser.add_argument("--scales", metavar="LIST", type=str,
                        default="[0.7, 1, 1.2]", help="List of scales")
    parser.add_argument("--flip", action="store_true",
                        help="Use horizontal flipping")
    parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                        default="final",
                        help="How the output files are formatted."
                        " -- palette: color coded predictions"
                        " -- raw: gray-scale predictions"
                        " -- prob: gray-scale predictions plus probabilities")
    parser.add_argument("data", metavar="IN_DIR",
                        type=str, help="Path to dataset")
    parser.add_argument("output", metavar="OUT_DIR", type=str,
                        help="Path to output folder")
    parser.add_argument("--world-size", metavar="WS", type=int,
                        default=1, help="Number of GPUs")

    # Load configuration
    args = parser.parse_args()

    # Torch stuff
    torch.cuda.set_device(args.rank)
    cudnn.benchmark = True

    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot(args.snapshot)
    model = SegmentationModule(body, head, 256, 65, args.fusion_mode)
    model.cls.load_state_dict(cls_state)
    model = model.cuda().eval()
    print(model)

    print("All done.")
