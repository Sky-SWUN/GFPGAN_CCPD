import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool


def process_task(imgs_path, size, target_folder):
    """剪切并保存正方形区域的图片
    """

    os.makedirs(target_folder, exist_ok=True)

    for img_path in tqdm(imgs_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (size, size))
        img = cv2.resize(img, (512, 512))
        # cv2.imwrite(os.path.join(target_folder, str(size)+"x"+str(size)+"_" + img_name.split(".")[0] + ".jpg"), img)
        cv2.imwrite(os.path.join(target_folder, os.path.basename(img_path)), img)


if __name__ == "__main__":

    imgs_dir = "custom_tools/BasicSR/datasets/ccpd/ccpd_crop_rect_PerspectiveTransform_front_1000_realesr_512"
    imgs_name = os.listdir(imgs_dir)
    imgs_path = [os.path.join(imgs_dir, name) for name in imgs_name]

    task_list = []

    # 将列表分为n个
    n = 8

    if len(imgs_name) % n == 0:
        capacity = int(len(imgs_name) / n)
    else:
        capacity = int((len(imgs_name) / n) + 1)

    for i in range(n):
        task_list.append(imgs_path[i * capacity:(i + 1) * capacity])

    # 创建进程
    p = Pool(n)

    for i in range(n):
        p.apply_async(
            process_task,
            args=(
                task_list[i],
                20,
                "datasets/20x20",
            ),
            error_callback=print,
        )

    p.close()
    p.join()
