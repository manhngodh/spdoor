import os

import cv2
import numpy as np
import tensorflow as tf

from model.livenessnet import LivenessNet

print(tf.version.VERSION)
backbone = 'mobilenetv3_small'
model = LivenessNet(backbone)
try:
    # latest = os.path.join('trained_models', backbone, 'cp-05.ckpt')
    #
    latest_path = os.path.join('trained_models', backbone)
    latest = tf.train.latest_checkpoint(latest_path)
    print('latest', latest)
    model.load_weights(latest)
except:
    raise Exception("No weight file!")


# image_path = 'data/test_224/spoof/download.png'
# img_preds = []
# img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
# # img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
#
# img_pred = (img / 255.0)
#
# img_preds.append(img_pred)
# img_preds = np.array(img_preds)

net_out_value = model.predict(np.random.rand(1, 224, 224, 3))
net_out_value = np.squeeze(net_out_value)
print(net_out_value)

model.save("saved_model/my_model")
# model.save('my_model.h5')
