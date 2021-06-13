import os

import cv2
import numpy as np
import pandas as pd

import config


def calculate_single_diff(img1_path, img2_path):  # calculate difference

    # read in two image sets
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    except AttributeError:
        print('wrong path for Image, please set up the right path')
        return 0

    if img2.shape != img1.shape:
        print('Wrong Image size, please choose the right image')
        return 0
    else:
        print('**************original bild value***********')
        print(img1)
        print('**************value after compression***********')
        print(img2)
        difference = img1 - img2
        print('**************difference***********')
        print(difference)
        save_to_excel(difference, 'Single_image_difference.xlsx')
        print('calculation complete, save the value to Single_image_difference.xlsx')
    return difference


def calculate_blocks_diff(dir1, dir2):  # calculate the Difference between the two given Image
    img_set1 = os.listdir(dir1)  # get all image from the first file
    img_set2 = os.listdir(dir2)  # get all image from the second file
    size = min(len(img_set1), len(img_set2))
    for i in range(size):
        img1 = cv2.imread('./blocks/block' + str(i) + '.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
        img2 = cv2.imread('./recon/recon' + str(i) + '.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
        difference = img1 - img2
        print('calcathe the difference of {}th block'.format(i))
        save_to_excel(difference, 'block_differences/block_' + str(i) + '_difference.xlsx')
    print('All Calculation done')
    return True


def save_to_excel(difference, file_name):  # save the difference as Excel, file path need be given
    pd_data = pd.DataFrame(difference)
    writer = pd.ExcelWriter(file_name)
    pd_data.to_excel(writer)
    writer.save()
    return True


# def file_name(file_dir):  # get all the file in given folder
#     files_path = []
#     for root, dirs, files in os.walk(file_dir):
#         files_path.append(files)
#     return files_path


if __name__ == '__main__':
    """Given is the difference between the original image and after compression"""
    print('begin to calculate')
    calculate_single_diff(config.imageGray, config.imageReconstruct)

    """Given is the difference between block from the original image and after compression, value will be saved unter Block Difference"""
    print('begin to calculate')
    calculate_blocks_diff(config.dirBlocks, config.dirRecon)
