import os
import glob
import random
import shutil

number_images = 50
root_folder = os.path.dirname(os.path.abspath(__file__))
category = 'live'
source_folder_path = 'tmp/' + category
destination_folder_path = 'data/test/' + category
# source_folder_path = 'D:\\Dataset\\spoof'
# destination_folder_path = 'D:\\solution_face_spoofing\\spoofing_\\data\\test\\spoof'
os.makedirs(destination_folder_path, exist_ok=True)
os.makedirs(destination_folder_path.replace('test', 'train'), exist_ok=True)

total = len(os.listdir(source_folder_path))
test_amount = 0.4 * total
for i in range(int(test_amount)):
    print(i)
    source_path = random.choice(glob.glob(source_folder_path + '\\*'))  # change dir name to whatever
    dest = shutil.move(source_path, destination_folder_path)

for file_path in glob.glob(source_folder_path + '\\*'):
    train_path = destination_folder_path.replace('test', 'train')
    dest = shutil.move(file_path, train_path)
