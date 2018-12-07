import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
# from skimage.measure import structural_similarity as ssim
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
    args = parser.parse_args()

    # Get Directories
    gt_paths = os.listdir(args.in_gt)
    fake_paths = os.listdir(args.in_fake)

    if not gt_paths == fake_paths:
        raise Exception('Paths do not contain the same images?')

    # Loop over images
    avg_mse = 0
    avg_ssim = 0
    for i in tqdm(range(len(gt_paths))):
        
        # Load
        gt = cv2.imread(str(args.in_gt) + gt_paths[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        fake = cv2.imread(str(args.in_fake) + fake_paths[i])
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)

        # resize gt to size of fake
        gt = cv2.resize(gt, (fake.shape[1], fake.shape[0]))

        m = mse(gt,fake);     
        avg_mse += m

        s = ssim(gt, fake, data_range=(fake.max()-fake.min()));   
        avg_ssim += s
        # print ("MSE: ", m, " SSIM: ", s)

        # Show (Debug)
        view_scale = 0.3
        cv2.imshow('fake', cv2.resize(fake, (0,0), fx=view_scale, fy=view_scale))        
        cv2.imshow('gt', cv2.resize(gt, (0,0), fx=view_scale, fy=view_scale))
        cv2.waitKey(0)

    avg_mse /= len(gt_paths)
    avg_ssim /= len(gt_paths)
    print ("AVG MSE: ", avg_mse, " AVG SSIM: ", avg_ssim)