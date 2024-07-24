import cv2
import itertools
import math
import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from matplotlib import image
from tkinter import N
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,normalize)
from tqdm import tqdm
import random


# 运动模糊函数
def motion_blur(img=None, degree=12, angle=90):

    image = np.array(img)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


class FFHQDegradationDataset(data.Dataset):

    def __init__(self):
        super(FFHQDegradationDataset, self).__init__()

        # 默认参数
        opt = {
            'dataroot_gt':
            "custom_tools/BasicSR/datasets/ccpd/ccpd_crop_rect_PerspectiveTransform_front_1000_realesr_512",  # 输入图片的路径
            "out_size": 512,  # 输出图片大小
            "blur_kernel_size": 41,  # 模糊卷积核尺寸
            "kernel_list": ['iso', 'aniso'],  # 卷积核类型列表
            "kernel_prob": [0.5, 0.5],  # 卷积核参数
            "blur_sigma": [0.1, 10],  # 模糊参数
            "downsample_range": [10, 25],  # 下采样率
            "noise_range": [0, 20],  # 噪声区间
            "jpeg_range": [60, 100],  # jpeg伪影参数
            "color_jitter_prob": 0.3,
            "color_jitter_pt_prob": 0.3,
            "color_jitter_shift": 20,
            "brightness": (0.5, 1.5),
            "contrast": (0.5, 1.5),
            "saturation": (0, 1.5),
            "hue": (-0.1, 0.1),
            "motion_degree": [10, 20],  # 运动强度
            "motion_angle": [0, 180],  # 运动方向
        }

        #
        # 基本设置
        self.gt_folder = opt['dataroot_gt']
        self.out_size = opt['out_size']
        self.paths = paths_from_folder(self.gt_folder)

        # 模糊
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.noise_range = opt['noise_range']

        # 下采样
        self.downsample_range = opt['downsample_range']

        # jpeg伪影
        self.jpeg_range = opt['jpeg_range']

        # 颜色抖动
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        self.color_jitter_shift /= 255.

        # 光照 对比度 饱和度
        self.brightness = opt["brightness"]
        self.contrast = opt["contrast"]
        self.saturation = opt["saturation"]
        self.hue = opt["hue"]

        # 运动模糊
        self.motion_degree = opt["motion_degree"]
        self.motion_angle = opt["motion_angle"]

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def __getitem__(self, index):

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_gt = cv2.imread(gt_path)
        h, w, _ = img_gt.shape

        # 模糊
        if self.blur_sigma is not None:
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        else:
            img_lq = img_gt

        # 下采样
        if self.downsample_range is not None:
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        img_lq = img_lq / 255

        # 噪声
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)

        # jpeg伪影
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # 还原尺寸
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        # if self.color_jitter_prob is not None:
        #     img_lq = self.color_jitter(img_lq, self.color_jitter_shift)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt / 255, img_lq], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        # if self.color_jitter_pt_prob is not None:

        #     brightness = self.brightness
        #     contrast = self.contrast
        #     saturation = self.saturation
        #     hue = self.hue

        #     img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # 取整和压缩值到某个区间
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # 张量转为图片格式
        lq = tensor2img(img_lq)
        gt = tensor2img(img_gt)

        # 运动模糊
        if self.motion_degree is not None and self.motion_angle is not None:
            lq = motion_blur(
                img=lq,
                degree=random.randint(self.motion_degree[0], self.motion_degree[1]),
                angle=random.randint(self.motion_angle[0], self.motion_angle[1]),
            )

        return {'lq': lq, 'gt': gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":

    dataset = FFHQDegradationDataset()

    dataset.gt_folder = "custom_tools/BasicSR/datasets/ccpd/ccpd_crop_rect_PerspectiveTransform_front_1000_realesr_512"
    dataset.downsample_range = None

    # blur_sigma = [[0.1, 1], [1, 10], [None]]
    # noise_range = [[0, 20], [20, 40], [None]]
    # jpeg_range = [[80, 120], [120, 160], [None]]
    # motion_degree = [[10, 15], [15, 20], [None]]
    # motion_angle = [[0, 45], [45, 90], [None]]

    # blur_sigma = [[0.1, 10], [None]]
    # noise_range = [[0, 40], [None]]
    # jpeg_range = [[80, 160], [None]]
    # motion_degree = [[10, 20], [None]]
    # motion_angle = [[0, 90], [None]]

    blur_sigma = [[19, 20], [None]]
    noise_range = [[79, 80], [None]]
    jpeg_range = [[300, 310], [None]]
    motion_degree = [[39, 40], [None]]
    motion_angle = [[0, 90], [None]]

    for i, result in enumerate(itertools.product(
            blur_sigma,
            noise_range,
            jpeg_range,
            motion_degree,
            motion_angle,
    )):

        # dataset.blur_sigma = result[0]
        # dataset.noise_range = result[1]
        # dataset.jpeg_range = result[2]
        # dataset.motion_degree = result[3]
        # dataset.motion_angle = result[4]

        # [None]转为None
        if not result[0] == [None]:
            dataset.blur_sigma = result[0]
        else:
            dataset.blur_sigma = result[0][0]

        if not result[1] == [None]:
            dataset.noise_range = result[1]
        else:
            dataset.noise_range = result[1][0]

        if not result[2] == [None]:
            dataset.jpeg_range = result[2]
        else:
            dataset.jpeg_range = result[2][0]

        # if not result[3] == [None]:
        #     dataset.motion_degree = result[3]
        # else:
        #     dataset.motion_degree = result[3][0]

        # if not result[4] == [None]:
        #     dataset.motion_angle = result[4]
        # else:
        #     dataset.motion_angle = result[4][0]

        if not result[3] == [None] and not result[4] == [None]:
            dataset.motion_degree = result[3]
            dataset.motion_angle = result[4]
        elif result[3]==result[4]==[None]:
            dataset.motion_degree = result[3][0]
            dataset.motion_angle = result[4][0]
        else:
            continue


        # 文件夹名字
        folder_name = "_".join(list(map(str, sum(list(result), []))))
        project_name = str(list(result)).replace("],","_").replace(",","-").replace(" ","").replace("[","").replace("]","")

        # 目标存放地址
        target_dir = os.path.join("./datasets", folder_name)

        # 只建立单一模糊文件夹
        # if result[3]==result[4]==[None]:
        #     if folder_name.count("None")==4:
        #         os.makedirs(target_dir, exist_ok=True)
        # elif not result[3]==[None] and not result[4]==[None]:
        #     if folder_name.count("None")==3:
        #         os.makedirs(target_dir, exist_ok=True)

        os.makedirs(target_dir, exist_ok=True)

        # 在单一文件夹下处理
        if os.path.exists(target_dir):

            # for imgs_info_dict in tqdm(dataset):
            #     cv2.imwrite(
            #         os.path.join(target_dir, os.path.basename(imgs_info_dict["gt_path"])),
            #         imgs_info_dict["lq"],
            #     )

            # print("python inference_gfpgan.py -i {} -o {}".format(target_dir,target_dir+"_result"))

            # print("python custom_tools/recognition.py -d {}".format(target_dir))
            # print("python custom_tools/recognition.py -d {}".format(target_dir+"_result"))

            # print("python custom_tools/SSIM_PSNR_MSE.py -t {}".format(target_dir+"_result"))
            print("python custom_tools/SSIM_PSNR_MSE.py -t {}".format(target_dir))

            # print(project_name)
