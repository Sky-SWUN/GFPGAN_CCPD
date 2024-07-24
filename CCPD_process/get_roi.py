"""
本脚本用于获取label和正方形车牌区域
"""

import os
from multiprocessing import Pool

import cv2
import numpy
from tqdm import tqdm
import torch.nn

def get_data_from_path(img_path):
    """输入CPDD图片路径返回标签字典
    """
    file_name = os.path.basename(img_path)
    all_data = file_name.split("-")

    label_dict = {
        "angle": all_data[1].split("_"),
        "box": all_data[2].replace("_", "&").split("&"),
        "points": all_data[3].replace("_", "&").split("&"),
        "content": all_data[4].split("_"),
    }

    for key, value in label_dict.items():
        label_dict[key] = list(map(int, value))

    label_dict.update({"path": img_path})

    return label_dict


def paint_info_on_img(label_dict, save_path):
    """将标签信息画在图片上
    """

    # 读取图片
    img = cv2.imread(label_dict["path"])

    # 画位置框
    cv2.rectangle(
        img,
        (label_dict["box"][0], label_dict["box"][1]),
        (label_dict["box"][2], label_dict["box"][3]),
        color=(0, 255, 255),
        thickness=3,
    )

    # 画边缘框
    pts = numpy.array([
        [label_dict["points"][0], label_dict["points"][1]],
        [label_dict["points"][2], label_dict["points"][3]],
        [label_dict["points"][4], label_dict["points"][5]],
        [label_dict["points"][6], label_dict["points"][7]],
    ], numpy.int32)  # 数据类型必须为 int32
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )

    cv2.imwrite(save_path, img)


def get_rectroi_img(label_dict, save_path):
    """获取正方形的车牌区域
    """

    #     (label_dict["box"][0], label_dict["box"][1]),
    # (label_dict["box"][2], label_dict["box"][3]),

    # 读取图片
    img = cv2.imread(label_dict["path"])

    # 设置切片
    xmin = min(
        label_dict["points"][0],
        label_dict["points"][2],
        label_dict["points"][4],
        label_dict["points"][6],
    )
    xmax = max(
        label_dict["points"][0],
        label_dict["points"][2],
        label_dict["points"][4],
        label_dict["points"][6],
    )
    ymin = min(
        label_dict["points"][1],
        label_dict["points"][3],
        label_dict["points"][5],
        label_dict["points"][7],
    )
    ymax = max(
        label_dict["points"][1],
        label_dict["points"][3],
        label_dict["points"][5],
        label_dict["points"][7],
    )
    y_expand = ((xmax - xmin) - (ymax - ymin)) // 2
    new_ymax = ymax + y_expand
    new_ymin = new_ymax - (xmax - xmin)

    # 保存图片
    if img[new_ymin:new_ymax, xmin:xmax].size>0:
        cv2.imwrite(
            os.path.join(save_path, os.path.basename(label_dict["path"])),
            img[new_ymin:new_ymax, xmin:xmax],
        )


def process_task(imgs_path):
    """剪切并保存正方形区域的图片
    """

    for img_name in tqdm(imgs_path):

        img_path = os.path.join("CCPD2019/ccpd_base", img_name)
        label_dict = get_data_from_path(img_path)
        get_rectroi_img(label_dict, "CCPD2019/ccpd_crop_rect")


if __name__ == "__main__":

    # imgs_path = os.listdir("CCPD2019/ccpd_base")
    # task_list = []

    # # 将列表分为n个
    # n = 8

    # if len(imgs_path) % n == 0:
    #     capacity = int(len(imgs_path) / n)
    # else:
    #     capacity = int((len(imgs_path) / n) + 1)

    # for i in range(n):
    #     task_list.append(imgs_path[i * capacity:(i + 1) * capacity])

    # # 创建进程
    # p = Pool(n)

    # for i in range(n):
    #     p.apply_async(
    #         process_task,
    #         args=(
    #             task_list[i],
    #         ),
    #         error_callback=print,
    #     )

    # p.close()
    # p.join()

    label_dict = get_data_from_path("./0227729885057-90_76-152&507_488&590-493&583_185&590_151&504_459&497-0_0_12_14_24_27_31-97-146.jpg")
    data = numpy.random.rand(1,3,5,5)
    paint_info_on_img(label_dict=label_dict,save_path="./test.jpg")
