import os

import cv2
import matplotlib
import numpy as np

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet
from model.mobilenetv3.mobilenetv3_factory import build_mobilenetv3
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from utils import plot_confusion_matrix

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

backbone = 'mobilenetv3_small'
model = LivenessNet(backbone)

try:
    # latest = os.path.join('trained_models', backbone, 'cp-05.ckpt')

    latest_path = os.path.join('trained_models', backbone)
    latest = tf.train.latest_checkpoint(latest_path)
    print('latest', latest)
    model.load_weights(latest)
except:
    raise Exception("No weight file!")


image_path = '../data/test_224/spoof/download.png'
img_preds = []
img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
img_pred = (img / 255.0)

img_preds.append(img_pred)
img_preds = np.array(img_preds)
net_out_value = model.predict(img_preds)
net_out_value = np.squeeze(net_out_value)
print(net_out_value)

