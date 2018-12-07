import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as ssim

import pdb


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


if __name__ == '__main__':

    # Load Arguments
    parser = argparse.ArgumentParser(
        description="Testing script for the Vistas segmentation model")

    parser.add_argument("--in-gt", type=str, default="folder_gt",
                        help="Path to folder containing sharp ground truth images.")
    parser.add_argument("--in-fake", type=str, default="folder_fake",
                        help="Path to folder containing deblurred images.")
    parser.add_argument("--in-blur", type=str, default="folder_fake",
                        help="Path to folder containing deblurred images.")
    args = parser.parse_args()

    # Get Directories
    gt_paths = os.listdir(args.in_gt)
    fake_paths = os.listdir(args.in_fake)
    blur_paths = os.listdir(args.in_blur)

    if not gt_paths == fake_paths:
        raise Exception('Paths do not contain the same images?')

    # Loop over images
    fake_avg_mse = 0
    fake_avg_ssim = 0
    blur_avg_mse = 0
    blur_avg_ssim = 0
    for i in tqdm(range(len(gt_paths))):
        
        # Load
        gt = cv2.imread(str(args.in_gt) + gt_paths[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        fake = cv2.imread(str(args.in_fake) + fake_paths[i])
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
        blur = cv2.imread(str(args.in_blur) + blur_paths[i])
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # resize gt to size of fake
        gt = cv2.resize(gt, (fake.shape[1], fake.shape[0]))
        blur = cv2.resize(blur, (fake.shape[1], fake.shape[0]))

        # gt vs fake
        m = mse(gt,fake);     
        fake_avg_mse += m
        s = ssim(gt, fake, data_range=(fake.max()-fake.min()));   
        fake_avg_ssim += s

        # gt vs blur
        m = mse(gt,blur);     
        blur_avg_mse += m
        s = ssim(gt, blur, data_range=(blur.max()-blur.min()));   
        blur_avg_ssim += s

        # Show (Debug)
        view_scale = 1
        cv2.imshow('gt', cv2.resize(gt, (0,0), fx=view_scale, fy=view_scale))
        cv2.imshow('fake', cv2.resize(fake, (0,0), fx=view_scale, fy=view_scale))        
        cv2.imshow('blur', cv2.resize(blur, (0,0), fx=view_scale, fy=view_scale))        
        cv2.waitKey(0)

    fake_avg_mse /= len(gt_paths)
    fake_avg_ssim /= len(gt_paths)
    print ("FAKE AVG MSE: ", fake_avg_mse, " FAKE AVG SSIM: ", fake_avg_ssim)

    blur_avg_mse /= len(gt_paths)
    blur_avg_ssim /= len(gt_paths)
    print ("BLUR AVG MSE: ", blur_avg_mse, " BLUR AVG SSIM: ", blur_avg_ssim)