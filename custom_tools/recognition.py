from ast import arg
import cv2
from hyperlpr import *
from tqdm import tqdm
import argparse

# 车牌字符字典
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


# 获取标签信息
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d',default="custom_tools/BasicSR/datasets/ccpd/ccpd_crop_rect_PerspectiveTransform_front_1000_realesr_512", type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    imgs_path = os.listdir(data_dir)

    # 字符精度
    character_accuracy = 6 * len(imgs_path)
    # 车牌识别精度
    global_accuracy = len(imgs_path)

    # 计算准确率
    for loc, img_path in enumerate(tqdm(imgs_path)):

        # 读取车牌图片
        image = cv2.imread(os.path.join(data_dir, img_path))

        # 获取标签
        real_characters = get_data_from_path(img_path)["content"][1:]

        # 识别结果
        result = HyperLPR_plate_recognition(image, charSelectionDeskew=False)

        # 计算损失函数
        if result:
            correct_num = 0
            for loc, character in enumerate(result[0][0][1:7]):
                # print(plate_dict[character.lower()] , real_characters[loc])
                if not '\u4e00' <= character <= '\u9fff':
                    if not plate_dict[character.lower()] == real_characters[loc]:
                        character_accuracy = character_accuracy - 1
                    else:
                        correct_num = correct_num + 1
                else:
                    character_accuracy = character_accuracy - 1

            if not correct_num == 6:
                global_accuracy = global_accuracy - 1
                # print(os.path.join(data_dir, img_path))
                # print(result[0][0])
        else:
            character_accuracy = character_accuracy - 6
            global_accuracy = global_accuracy - 1

    print(args.data_dir)
    print("字符准确率为：{:.1f}%".format(character_accuracy / (6 * 1000) * 100))
    print("识别准确率为：{:.1f}%".format(global_accuracy / 1000 * 100))
