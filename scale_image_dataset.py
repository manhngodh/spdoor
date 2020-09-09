import glob
import multiprocessing
import os

import cv2

IMG_WIDTH = 224
IMG_HEIGHT = 224

source_data_folder = 'data/'
destination_folder_scaled = 'data/'
train_test_sets = ['test', 'train']
categories = ['live', 'spoof']


def resize_image_dataset(file_path):
    # Read
    image = cv2.imread(file_path)

    # Resize
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT))

    file_name = os.path.basename(file_path)
    normpath = os.path.normpath(file_path)
    sep_directories = normpath.split(os.sep)
    # folder_name = os.path.basename(os.path.dirname(file_path))
    # print(file_name, folder_name)
    dst_path = os.path.join(destination_folder_scaled, f'{sep_directories[-3]}_{IMG_WIDTH}x{IMG_HEIGHT}', f'{sep_directories[-2]}')
    print(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    cv2.imwrite(os.path.join(dst_path, file_name), image)


if __name__ == '__main__':
    cpu = multiprocessing.cpu_count()
    for set_name in train_test_sets:
        for item in categories:
            path_names = [file for file in glob.glob(os.path.join(source_data_folder, set_name, item, "*"))]

            # print(path_names[1:5], len(path_names))
            # print(cpu)
            sub_sample_number = int(len(path_names) / cpu)

            # list_sub = [path_names[x:x + sub_sample_number] for x in range(0, len(path_names), sub_sample_number)]
            # for sub in list_sub:
            #     print(len(sub))
            # print(len(list_sub))
            with multiprocessing.Pool(cpu) as pool:
                pool.map(resize_image_dataset, path_names)
