import glob
import os

import numpy as np
import cv2
import tensorflow as tf
import time

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet


class TestLiveness:
    def __init__(self):
        backbone = 'mobilenetv3_small'
        self.model = LivenessNet(backbone)

        try:
            # latest = os.path.join('trained_models', backbone, 'cp-05.ckpt')

            latest_path = os.path.join('trained_models', backbone)
            latest = tf.train.latest_checkpoint(latest_path)
            print('latest', latest)
            self.model.load_weights(latest)
        except:
            raise Exception("No weight file!")

    def run_(self, video_path, true_label):
        print(video_path, true_label)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            # Capture frame-by-frame
            try:
                ret, frame = cap.read()

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img_pred = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img_pred = (img_pred / 255.0)

                img_pred = np.expand_dims(img_pred, axis=0)
                net_out_value = self.model.predict(img_pred)
                if net_out_value[0][0] > 0.5:
                    predicted_label = 'spoof'
                else:
                    predicted_label = 'live'

                if true_label != predicted_label:
                    print(net_out_value, 'wrong prediction')
                    cv2.imwrite(f'{destination_data_folder}/{true_label}/{round(time.time() * 1000)}.jpg', frame)
            except Exception as e:
                print(e)
                break


train_test_sets = ['live', 'spoof']
source_data_folder = 'data/videos'
destination_data_folder = 'tmp'

if __name__ == '__main__':
    test_liveness = TestLiveness()
    for label in train_test_sets:
        os.makedirs(os.path.join(destination_data_folder, label), exist_ok=True)
        path_names = [file for file in glob.glob(os.path.join(source_data_folder, label, "*"))]
        for path in path_names:
            test_liveness.run_(path, label)
