import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse

from segmentation_module import SegmentationModule, load_snapshot, flip
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset import VEGANSegmentationDataset, vegan_segmentation_collate
from dataset.transform import SegmentationTransform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Testing script for the Vistas segmentation model")
    parser.add_argument("--snapshot", type=str, default="weights/wide_resnet38_deeplab_vistas.pth",
                        help="Snapshot file to load")
    parser.add_argument("--in-gt", type=str, default="folder_gt",
                        help="Path to folder containing sharp ground truth images.")
    parser.add_argument("--in-fake", type=str, default="folder_fake",
                        help="Path to folder containing deblurred images.")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Path to output folder.")
    parser.add_argument("--name", type=str, default="ganisgood",
                        help="Experiment folder name to be used in output folder.")
    # minimally needed
    parser.add_argument("--rank", metavar="RANK",
                        type=int, default=0, help="GPU id")
    parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                        help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
    parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                        default="final",
                        help="How the output files are formatted."
                        " -- palette: color coded predictions"
                        " -- raw: gray-scale predictions"
                        " -- prob: gray-scale predictions plus probabilities")
    # not sure uses below here yet
    parser.add_argument("--scales", metavar="LIST", type=str,
                        default="[0.7, 1, 1.2]", help="List of scales")
    parser.add_argument("--flip", action="store_true",
                        help="Use horizontal flipping")
    parser.add_argument("--world-size", metavar="WS", type=int,
                        default=1, help="Number of GPUs")

    # Load configuration
    args = parser.parse_args()

    # experiment setup
    # check if output directory exists, if not, then make
    # check if name is there, if not, then make
    # create a logdir for tensorboard
    # create all paths

    # Torch stuff
    torch.cuda.set_device(args.rank)
    cudnn.benchmark = True

    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot(args.snapshot)
    model = SegmentationModule(body, head, 256, 65, args.fusion_mode)
    model.cls.load_state_dict(cls_state)
    model = model.cuda().eval()
    print(model)

    # Create data loaders
    transformation = SegmentationTransform(
        2048,
        (0.41738699, 0.45732192, 0.46886091),
        (0.25685097, 0.26509955, 0.29067996),
    )
    dataset = VEGANSegmentationDataset(
        args.in_gt, args.in_fake, transformation)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        sampler=DistributedSampler(dataset, args.world_size, args.rank),
        num_workers=2,
        collate_fn=vegan_segmentation_collate,
        shuffle=False
    )

    print(len(data_loader))
    print("All done.")
