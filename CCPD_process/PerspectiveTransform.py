"""
本脚本用于透视变换矫正数据
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

from get_roi import get_data_from_path

# os.makedirs("CCPD2019/ccpd_crop_rect_front_50000_realesr_results_512_PerspectiveTransform",exist_ok=True)

for img_name in tqdm(os.listdir("test_data")):
    # img_path = "./CCPD2019/ccpd_base/" + img_name
    img_path = "test_data/0227729885057-90_76-152&507_488&590-493&583_185&590_151&504_459&497-0_0_12_14_24_27_31-97-146.jpg"
    info = get_data_from_path(img_path=img_path)
    img = cv2.imread(info["path"])

    # 取出原图上的角点位置
    original_points = np.array(
        [
            [info["points"][0], info["points"][1]],
            [info["points"][2], info["points"][3]],
            [info["points"][4], info["points"][5]],
            # [info["points"][6], info["points"][7]],
        ],
        dtype="float32",
    )

    # 设置新的角点位置
    new_points = np.array(
        [
            [440, 290],
            [0, 290],
            [0, 150],
            # [440, 150],
        ],
        dtype="float32",
    )

    # 设置转换矩阵
    # 透视变换
    # M = cv2.getPerspectiveTransform(original_points, new_points)
    # 仿射变换
    M = cv2.getAffineTransform(original_points, new_points)

    # 透视变换
    # out_img = cv2.warpPerspective(img, M, (440, 440))
    # 仿射变换
    out_img = cv2.warpAffine(img, M, (440, 440))

    # resize透视变换后的图片
    result_img = cv2.resize(out_img, (512, 512))

    cv2.imwrite("test.jpg", result_img)

    # 保存resize后的图片
    cv2.imwrite(
        os.path.join(
            "CCPD2019/ccpd_crop_rect_front_50000_realesr_results_512_PerspectiveTransform",
            os.path.basename(img_path),
        ), result_img)

    # 保存并展示每个字符的位置
    # cv2.imwrite("1.jpg", result_img[175:337, 10:75])
    # cv2.imwrite("2.jpg", result_img[175:337, 75:140])
    # cv2.imwrite("3.jpg", result_img[175:337, 165:230])
    # cv2.imwrite("4.jpg", result_img[175:337, 230:295])
    # cv2.imwrite("5.jpg", result_img[175:337, 295:360])
    # cv2.imwrite("6.jpg", result_img[175:337, 360:425])
    # cv2.imwrite("7.jpg", result_img[175:337, 437:502])