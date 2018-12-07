import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from segmentation_module import SegmentationModule, load_snapshot, flip
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset import VEGANSegmentationDataset, vegan_segmentation_collate
from dataset.transform import SegmentationTransform
from palette_module import get_pred_image


def confusion_to_iou(confusion):
    denom = confusion.sum(axis=1) + confusion.sum(axis=0) - \
        np.diag(confusion).astype(np.float32)
    numerator = np.diag(confusion).astype(np.float32)
    # stop divide by zero!
    numerator = numerator[denom != 0]
    denom = denom[denom != 0]
    iou = numerator / denom
    return np.nanmean(iou)


def save_palette_image(output_path, probs, preds, rec, gt=True):
    # saving palettes
    for i, (prob, pred) in enumerate(zip(torch.unbind(probs, dim=0), torch.unbind(preds, dim=0))):
        out_size = rec["meta"][i]["size"]
        img_name = rec["meta"][i]["idx"]

        # Save prediction
        prob = prob.cpu()
        pred = pred.cpu()
        pred_img = get_pred_image(pred, out_size, True)
        if gt:
            pred_img.save(os.path.join(output_path,
                                       img_name + "_gt" + ".png"))
        else:
            pred_img.save(os.path.join(output_path,
                                       img_name + "_fake" + ".png"))


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
    parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max", "iou_max"], default="iou_max",
                        help="How to fuse the outputs. Options: 'mean', 'voting', 'max', 'iou_max'")
    parser.add_argument("--scales", metavar="LIST", type=str,
                        default="[1]", help="List of scales")
    # minimally needed
    parser.add_argument("--rank", metavar="RANK",
                        type=int, default=0, help="GPU id")
    parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                        default="final",
                        help="How the output files are formatted."
                        " -- palette: color coded predictions"
                        " -- raw: gray-scale predictions"
                        " -- prob: gray-scale predictions plus probabilities")
    # not sure uses below here yet
    parser.add_argument("--flip", action="store_true",
                        help="Use horizontal flipping")
    parser.add_argument("--world-size", metavar="WS", type=int,
                        default=1, help="Number of GPUs")

    # Load configuration
    args = parser.parse_args()

    # experiment setup
    out_path = args.out_dir
    if not os.path.exists(out_path):
        os.makedirs(args.out_dir)
    experiment_path = os.path.join(out_path, args.name)
    image_output_path = os.path.join(experiment_path, "out_images")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(image_output_path)
    elif not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    # Torch stuff
    torch.cuda.set_device(args.rank)
    cudnn.benchmark = True

    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot(args.snapshot)
    model = SegmentationModule(body, head, 256, 65, args.fusion_mode)
    model.cls.load_state_dict(cls_state)
    model = model.cuda().eval()
    print(model)
    print('\n\n')

    # Create data loaders
    transformation = SegmentationTransform(
        512,
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

    MSEloss = torch.nn.modules.loss.MSELoss()
    total_MSE_loss = 0

    # progress bar stuff
    scales = eval(args.scales)
    dataset_confusion = np.zeros((65, 65), dtype=np.int32)
    with torch.no_grad():
        with tqdm(data_loader, desc="IOU") as t:
            for batch_i, data in enumerate(t):
                # print(
                #     "Testing batch [{:3d}/{:3d}]".format(batch_i + 1, len(data_loader)))

                # get the data
                gt_img = data["gt_img"].cuda(non_blocking=True)
                fake_img = data["fake_img"].cuda(non_blocking=True)

                if gt_img.shape != fake_img.shape:
                    fake_img = functional.interpolate(fake_img,
                                                      (gt_img.shape[2],
                                                       gt_img.shape[3]),
                                                      mode='bilinear',
                                                      align_corners=None)

                total_MSE_loss += MSEloss(fake_img, gt_img)

                # inference
                gt_probs_img, gt_preds_img, gt_logits = model(
                    gt_img, scales, args.flip)
                fake_probs_img, fake_preds_img, fake_logits = model(
                    fake_img, scales, args.flip)

                # getting the classes
                gt_probs = functional.softmax(gt_logits, dim=1)
                _, gt_cls = gt_probs.squeeze().max(0)
                gt_cls = gt_cls.data.cpu().numpy().reshape(-1)
                fake_probs = functional.softmax(fake_logits, dim=1)
                _, fake_cls = fake_probs.squeeze().max(0)
                fake_cls = fake_cls.data.cpu().numpy().reshape(-1)

                # construct confusion matrix
                confusion = np.zeros((65, 65), dtype=np.int32)
                for i in range(gt_cls.shape[0]):
                    confusion[gt_cls[i], fake_cls[i]] += 1
                dataset_confusion += confusion

                # handling output images
                save_palette_image(image_output_path,
                                   gt_probs_img, gt_preds_img, data, gt=True)
                save_palette_image(image_output_path,
                                   fake_probs_img, fake_preds_img, data, gt=False)

                # update tqdm
                t.set_description("<IOU: %f>" %
                                  confusion_to_iou(confusion))
                t.refresh()

    # find dataset iou
    print("TEST DATASET IOU: {}".format(confusion_to_iou(dataset_confusion)))
    print("TEST DATA MSE LOSS: {}".format(total_MSE_loss/len(dataset)))

    # save images?
    # TODO: only plot the relevant classes in the confusion matrix
    plt.imshow(dataset_confusion)
    plt.show()
    print("All done.")
