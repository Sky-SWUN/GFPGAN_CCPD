"""
本脚本用于测试车牌识别的损失添加
"""

from hyperlpr import *
import cv2

# CCPD字符词典
plate_dict = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "j": 8,
    "k": 9,
    "l": 10,
    "m": 11,
    "n": 12,
    "p": 13,
    "q": 14,
    "r": 15,
    "s": 16,
    "t": 17,
    "u": 18,
    "v": 19,
    "w": 20,
    "x": 21,
    "y": 22,
    "z": 23,
    "0": 24,
    "1": 25,
    "2": 26,
    "3": 27,
    "4": 28,
    "5": 29,
    "6": 30,
    "7": 31,
    "8": 32,
    "9": 33
}

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

def get_loss_from_recognition(img_path):

    # 读取图片
    image = cv2.imread(img_path)

    # 获取标签
    real_characters = get_data_from_path(img_path)["content"][1:]

    # 识别结果
    result = HyperLPR_plate_recognition(image, charSelectionDeskew=False)

    # 设置损失函数
    loss = 6

    # 计算损失函数
    if result:
        for loc,character in enumerate(result[0][0][1:]):
            if plate_dict[character.lower()] == real_characters[loc]:
                loss = loss-1

get_loss_from_recognition("01-90_87-240&501_441&563-437&567_237&561_230&490_430&496-0_0_2_12_32_26_26-180-20.jpg")