import os
import shutil

import cv2
import matplotlib
import numpy as np

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet
from model.mobilenetv3.mobilenetv3_factory import build_mobilenetv3
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from utils import plot_confusion_matrix

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

backbone = 'mobilenetv3'

if backbone == 'mobilenetv2':
    model = LivenessNet(backbone)
elif backbone == 'mobilenetv3':
    model = build_mobilenetv3(
        "large",
        input_shape=(224, 224, 3),
        num_classes=2,
        width_multiplier=1.0,
    )
else:
    raise Exception('please choose a model!!!!!!!!!!')

try:
    # latest_path = os.path.join('trained_models', backbone)
    # checkpoint = tf.train.latest_checkpoint(latest_path)
    # print('latest', latest)
    specific_checkpoint_path = os.path.join('trained_models', backbone, 'good_checkpoint', 'cp-05.ckpt')
    # checkpoint = tf.train.load_checkpoint(specific_checkpoint_path)
    model.load_weights(specific_checkpoint_path)
except:
    raise Exception("No weight file!")

PATH = '../data'
# PATH = os.path.join('dataset_wis/train_224', category)

unspecified_folder = os.path.join(PATH, 'unspecify')
new_folder = os.path.join(PATH, 'specified')

for file_name in os.listdir(unspecified_folder):
    image_path = os.path.join(unspecified_folder, file_name)
    print(image_path)
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # img = cv2.imread('/media/data/manh-nx/spoofing_/dataset_wis/train/live/img_1576811846020.jpg', cv2.COLOR_BGR2RGB)

    # img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_pred = (img_pred / 255.0)

    img_pred = np.expand_dims(img_pred, axis=0)
    net_out_value = model.predict(img_pred)
    net_out_value = np.argmax(net_out_value, axis=1)
    if net_out_value[0] == 0:
        category_folder = 'live'
    else:
        category_folder = 'spoof'

    source_path = image_path
    # Destination path
    category_path = os.path.join(new_folder, category_folder)
    os.makedirs(category_path, exist_ok=True)
    destination_path = os.path.join(category_path, file_name)

    # Move the content of
    # source to destination
    dest = shutil.move(source_path, destination_path)
    print(destination_path)
