import os

import numpy as np
import cv2
import tensorflow as tf
import time

from config import IMG_HEIGHT, IMG_WIDTH
from model.livenessnet import LivenessNet

if __name__ == '__main__':

    cap = cv2.VideoCapture('data/videos/3.mp4')

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

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pred = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img_pred = (img_pred / 255.0)

        img_pred = np.expand_dims(img_pred, axis=0)
        net_out_value = model.predict(img_pred)
        if net_out_value[0][0] > 0.5:
            print(net_out_value, 'spoof')
            cv2.imwrite(f'tmp/{round(time.time()*1000)}.jpg', frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
