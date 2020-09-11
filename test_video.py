import glob
import os
import shutil

import numpy as np
import cv2
import tensorflow as tf
import time

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet
from utils.detections import pad_input_image, recover_pad_output, load_yaml


class TestLiveness:
    def __init__(self):
        backbone = 'mobilenetv3_small'
        self.model = LivenessNet(backbone)
        self.cfg = load_yaml('./configs/retinaface_mbv2.yaml')
        self.detection_model = tf.saved_model.load('saved_model/detection_model')
        self.model = tf.keras.models.load_model('saved_model/spoof_model')
        # try:
        #     # latest = os.path.join('trained_models', backbone, 'cp-05.ckpt')
        #
        #     latest_path = os.path.join('trained_models', backbone)
        #     latest = tf.train.latest_checkpoint(latest_path)
        #     print('latest', latest)
        #     self.model.load_weights(latest)
        # except:
        #     raise Exception("No weight file!")

    def run_(self, video_path, true_label):
        print(video_path, true_label)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            # Capture frame-by-frame
            try:
                ret, frame = cap.read()

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = np.float32(img.copy())

                img_detection, pad_params = pad_input_image(img, max_steps=max(self.cfg['steps']))

                # run model
                outputs = self.detection_model(img_detection[np.newaxis, ...]).numpy()

                # recover padding effect
                outputs = recover_pad_output(outputs, pad_params)
                # print(len(outputs))
                if len(outputs) != 1:
                    # print('no face')
                    continue

                img_pred = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                # cv2.imwrite(f'x1/{round(time.time() * 1000)}.jpg', cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR))
                img_pred = (img_pred / 255.0)

                img_pred = np.expand_dims(img_pred, axis=0)
                net_out_value = self.model.predict(img_pred)
                if net_out_value[0][0] > 0.5:
                    predicted_label = 'spoof'
                else:
                    predicted_label = 'live'
                print(predicted_label)
                if true_label != predicted_label:
                    print(net_out_value, 'wrong prediction')
                    cv2.imwrite(f'{destination_data_folder}/{true_label}/{round(time.time() * 1000)}.jpg', frame)

                cv2.imshow('Windows', frame)
                if cv2.waitKey(1) == 27:
                    exit(0)
            except Exception as e:
                print(e)
                break


train_test_sets = ['spoof']
source_data_folder = 'data/videos'
destination_data_folder = 'tmp'

if __name__ == '__main__':
    test_liveness = TestLiveness()
    shutil.rmtree('tmp/live', ignore_errors=True)
    shutil.rmtree('tmp/spoof', ignore_errors=True)
    os.makedirs('tmp/live')
    os.makedirs('tmp/spoof')
    for label in train_test_sets:
        os.makedirs(os.path.join(destination_data_folder, label), exist_ok=True)
        # path_names = [file for file in glob.glob(os.path.join(source_data_folder, label, "*"))]
        path_names = [
            'data/videos/spoof/8.mp4'
        ]
        for path in path_names:
            test_liveness.run_(path, label)
