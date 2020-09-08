from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

from config import IMG_WIDTH, IMG_HEIGHT


class LivenessNet(Model):
    def __init__(self, backbone='vgg16', output_bias=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if backbone == 'vgg16':
            self.backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
        elif backbone == 'mobilenetv2':
            self.backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet')
        elif backbone == 'mobilenetv3_large':
            from model.mobilenetv3_new.mobilenet_v3_large import MobileNetV3_Large
            self.backbone = MobileNetV3_Large((IMG_HEIGHT, IMG_WIDTH, 3), 1, include_top=False).build(plot=True)
        elif backbone == 'mobilenetv3_small':
            from model.mobilenetv3_new.mobilenet_v3_small import MobileNetV3_Small
            self.backbone = MobileNetV3_Small((IMG_HEIGHT, IMG_WIDTH, 3), 1, include_top=False).build(plot=True)
        elif backbone == 'resnet50':
            self.backbone = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
        else:
            raise Exception('must define backbone')
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.backbone.layers))

        # Fine tune from this layer onwards
        fine_tune_at = 100
        #
        # Freeze all the layers before the `fine_tune_at` layer
        # for layer in self.backbone.layers[:fine_tune_at]:
        #     layer.trainable = False

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        self.flatten = Flatten(name='flatten')
        # self.d1 = Dense(16, activation='relu', name='fc1')
        # self.d2 = Dense(1024, activation='relu', name='fc2')
        self.d3 = Dense(1, activation='sigmoid', bias_initializer=output_bias)

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.flatten(x)
        # x = self.d1(x)
        # x = Dropout(0.2)(x)
        # x = self.d2(x)
        # x = Dropout(0.5)(x)
        x = self.d3(x)
        return x


if __name__ == '__main__':
    model = LivenessNet()
    model.summary()
