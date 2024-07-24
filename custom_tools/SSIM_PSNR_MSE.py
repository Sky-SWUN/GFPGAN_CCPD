import os
from tqdm import tqdm
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', '-t',default="datasets/20x20", type=str)
    args = parser.parse_args()

    test_dir = args.test_dir

    generat_imgs_list = os.listdir(test_dir)

    psnr = []
    ssim = []
    mse = []

    for img_name in tqdm(generat_imgs_list):
        generat_img_path = os.path.join(test_dir,img_name)
        true_img_path = os.path.join("custom_tools/BasicSR/datasets/ccpd/ccpd_crop_rect_PerspectiveTransform_front_1000_realesr_512",img_name)

        img1 = cv2.imread(true_img_path)
        img2 = cv2.imread(generat_img_path)

        psnr.append(compare_psnr(img1, img2))
        ssim.append(compare_ssim(img1, img2, multichannel=True))  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
        mse.append(compare_mse(img1, img2))

    print(args.test_dir)
    print('PSNR:{}\nSSIM:{}\nMSE:{}'.format(sum(psnr)/1000, sum(ssim)/1000, sum(mse)/1000))

