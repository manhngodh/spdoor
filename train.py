import os

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCH
from metrics import METRICS
from model.livenessnet import LivenessNet

import ssl
import tensorflow as tf

print(tf.version.VERSION)

ssl._create_default_https_context = ssl._create_unverified_context

# Load du lieu
print("Load Data....")
# Load data
PATH = 'dataset_wis'
PATH2 = 'data'

train_dir = os.path.join(PATH2, 'train_224')
validation_dir = os.path.join(PATH2, 'test_224')

train_live_dir = os.path.join(train_dir, 'live')  # directory with our training live pictures
train_spoof_dir = os.path.join(train_dir, 'spoof')  # directory with our training spoof pictures
validation_live_dir = os.path.join(validation_dir, 'live')  # directory with our validation live pictures
validation_spoof_dir = os.path.join(validation_dir, 'spoof')  # directory with our validation spoof pictures

num_live_tr = len(os.listdir(train_live_dir))
num_spoof_tr = len(os.listdir(train_spoof_dir))
neg = num_live_tr
pos = num_spoof_tr

num_live_val = len(os.listdir(validation_live_dir))
num_spoof_val = len(os.listdir(validation_spoof_dir))

total_train = num_live_tr + num_spoof_tr
total_val = num_live_val + num_spoof_val

print('total training live images:', num_live_tr)
print('total training spoof images:', num_spoof_tr)

print('total validation live images:', num_live_val)
print('total validation spoof images:', num_spoof_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total_train, pos, 100 * pos / total_train))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    # brightness_range=[-0.2, 0.2],
    rotation_range=0.1,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(batch_size=BATCH_SIZE,
                                                    directory=train_dir,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=validation_dir,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='binary',
                                                        )

# image_batch, label_batch = next(train_generator)
#
# class_names = list(train_generator.class_indices.keys())
# print('label', label_batch, class_names)
#
# show_batch_binary(image_batch, label_batch, class_names)

#######################################################################################################################
backbone = 'mobilenetv3_small'

model = LivenessNet(backbone)
#
# initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
# model.save_weights(initial_weights)

# initialize the optimizer and compile the model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=METRICS)

model.build((None, 224, 224, 3))
model.summary()

for layer in model.layers:
    print(layer.trainable)

try:
    model_path = os.path.join('trained_models', backbone)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    # checkpoint_path = os.path.join('trained_models', backbone, 'good_checkpoint', 'cp-14.ckpt')
    print('latest', checkpoint_path)
    model.load_weights(checkpoint_path)
except:
    print("No weight file!")

# model.load_weights(initial_weights)
# model.layers[-1].bias.assign([0.0])

# weight_for_0 = (1 / neg) * total_train / 2.0
# weight_for_1 = (1 / pos) * total_train / 2.0
#
# class_weight = {0: weight_for_0, 1: weight_for_1}

checkpoint_path = os.path.join('trained_models', backbone, 'cp-{epoch:02d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_weights_only=True,
                                                save_best_only=True, verbose=1, period=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=EPOCH,
    callbacks=[checkpoint],
    validation_data=validation_generator,
    validation_steps=total_val // BATCH_SIZE
)

# x, y = zip(*(validation_generator[i] for i in range(len(validation_generator))))
# x_val = np.vstack(x)
# y_val = np.vstack(map(to_categorical, y))[:, 1]
#
# # image_batch, label_batch = next(validation_generator)
# # test_predictions_baseline = model.predict(image_batch)
# baseline_results = model.evaluate(x_val, y_val, verbose=1)
# for name, value in zip(model.metrics_names, baseline_results):
#     print(name, ': ', value)
# print()

# number_of_examples = len(validation_generator.filenames)
# number_of_generator_calls = math.ceil(number_of_examples / (1.0 * BATCH_SIZE))
# test_labels = []
#
# for i in range(0, int(number_of_generator_calls)):
#     test_labels.extend(np.array(validation_generator[i][1]))
# plot_cm(test_labels, test_predictions_baseline)

# Save model
# try:
#     model_path = os.path.join('trained_models', backbone)
#     checkpoint_path = tf.train.latest_checkpoint(model_path)
#     # checkpoint_path = os.path.join('trained_models', backbone, 'good_checkpoint', 'cp-14.ckpt')
#     print('latest', checkpoint_path)
#     model.load_weights(checkpoint_path)
# except:
#     print("No weight file!")
# model.save("saved_model/spoof_model")
#
# print("Finish model!")
