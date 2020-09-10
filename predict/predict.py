import glob
import os

import cv2
import matplotlib
import numpy as np

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from utils.utils import plot_confusion_matrix

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

backbone = 'mobilenetv3_small'

model = LivenessNet(backbone)

try:
    # latest = os.path.join('trained_models', backbone, 'cp-95.ckpt')
    latest = tf.train.latest_checkpoint(os.path.join('trained_models', backbone))
    print('latest', latest)
    model.load_weights(latest)
except:
    raise Exception("No weight file!")

# PATH = os.path.join('dataset_wis/train_224', category)

test_dir = '/media/data/manh-nx/spoofing_/data/test/live'
img_preds = []
for image_path in glob.glob(test_dir + '/*'):
    # image_path = os.path.join(test_dir, image_name)
    print(image_path)
    img_pred = cv2.imread(image_path)
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    # img = cv2.imread('/media/data/manh-nx/spoofing_/dataset_wis/train/live/img_1576811846020.jpg', cv2.COLOR_BGR2RGB)

    # img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (IMG_HEIGHT, IMG_WIDTH))
    img_pred = (img_pred / 255.0)

    img_preds.append(img_pred)
img_preds = np.array(img_preds)
net_out_value1 = model.predict(img_preds)
# net_out_value = np.squeeze(net_out_value)
# print(net_out_value1)


# net_out_value = np.argmax(net_out_value, axis=1) # for category classification
net_out_value = np.rint(net_out_value1)
# test_labels[:, 0] = 0
# print(net_out_value)

if 'live' in test_dir:
    test_labels = np.zeros_like(net_out_value)
else:
    test_labels = np.ones_like(net_out_value)

cm = confusion_matrix(test_labels, net_out_value)

cm_plt_labels = ['live', 'spoof']


plot_confusion_matrix(cm, cm_plt_labels)
matplotlib.use('agg')
plt.savefig('confusion.jpg')

# value_need = net_out_value[:, 1]
# n, bins, patches = plt.hist(value_need, density=True, facecolor='g', alpha=0.75)
# plt.xlabel('Numbers')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.grid(True)
# plt.savefig('preview2.jpg')
