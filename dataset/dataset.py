from itertools import chain

import glob
import torch
from PIL import Image
from os import path
from torch.utils.data import Dataset


class VEGANSegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, gt_path, fake_path, transform):
        super(VEGANSegmentationDataset, self).__init__()

        self.gt_path = gt_path
        self.fake_path = fake_path
        self.transform = transform

        # find all images in gt folder
        self.images = []
        for img_path in chain(*(glob.iglob(path.join(self.gt_path, ext)) for ext in VEGANSegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            test = path.join(self.fake_path, name_with_ext)
            self.images.append({
                "idx": idx,
                "gt_path": img_path,
                "fake_path": path.join(self.fake_path, name_with_ext)
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load gt image
        with Image.open(self.images[index]["gt_path"]) as gt_img_raw:
            # Load fake image
            fake_img_raw = Image.open(self.images[index]["fake_path"])
            size = gt_img_raw.size
            gt_img = self.transform(gt_img_raw.convert(mode="RGB"))
            fake_img = self.transform(fake_img_raw.convert(mode="RGB"))

        return {"gt_img": gt_img,
                "fake_img": fake_img,
                "meta": {"idx": self.images[index]["idx"], "size": size}}


def vegan_segmentation_collate(items):
    gt_imgs = torch.stack([item["gt_img"] for item in items])
    fake_imgs = torch.stack([item["fake_img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"gt_img": gt_imgs, "fake_img": fake_imgs, "meta": metas}


class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for img_path in chain(*(glob.iglob(path.join(self.in_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas}
