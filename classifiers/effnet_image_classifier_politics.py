# efficientnetv2-m for classifying images as one of the political person class

import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import transform
import os
import pandas as pd

img_size = (480, 480)
batch_size = 2


# load data for training and validation
def load_data(path):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    train_data = tf.keras.preprocessing.image_dataset_from_directory(path, validation_split=.20, subset='training',
                                                                     label_mode="categorical", seed=11,
                                                                     image_size=img_size, batch_size=1)
    class_names = tuple(train_data.class_names)
    train_size = train_data.cardinality().numpy()
    train_data = train_data.unbatch().batch(batch_size)
    train_data = train_data.repeat()
    train_data = train_data.map(lambda images, labels: (tf.keras.Sequential([normalization_layer])(images), labels))

    val_data = tf.keras.preprocessing.image_dataset_from_directory(path, validation_split=.20, subset='validation',
                                                                   label_mode="categorical", seed=11,
                                                                   image_size=img_size,
                                                                   batch_size=1)

    val_size = val_data.cardinality().numpy()
    val_data = val_data.unbatch().batch(batch_size)
    val_data = val_data.map(lambda images, labels: (normalization_layer(images), labels))

    return class_names, train_data, train_size, val_data, val_size


# execute training using the dataset and the class names and save the model
def execute_training(class_names, train_data, train_size, val_data, val_size):
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=img_size + (3,)), hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2", trainable=True),
                                 tf.keras.layers.Dropout(rate=0.2), tf.keras.layers.Dense(len(class_names),
                                                                                          kernel_regularizer=tf.keras.regularizers.l2(
                                                                                              0.0001))])
    model.build((None,) + img_size + (3,))
    print(model.summary())

    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc')
    ]

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                  metrics=[METRICS])

    steps_per_epoch = train_size // batch_size
    validation_steps = val_size // batch_size

    model.fit(
        train_data,
        epochs=10, steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps)

    saved_model_path = "../weights/model_effnet_politics_10epochs.h5"
    model.save(saved_model_path)


# train the model if not already trained and or if it does not have weights
def train_model():
    ds_path = '../data/dataset_csvs'

    # skip training if weights already present
    ckpt_file = Path("../weights/model_effnet_politics_10epochs.h5")
    if not ckpt_file.is_file():
        class_names, td, ts, vd, vs = load_data(ds_path)
        execute_training(class_names, td, ts, vd, vs)
    else:
        print('Weights for effnet on political figures already present!')


# predict model using saved weights on politics dataset and predict file path  and returns dataframe of outputs
def predict_model(filepath):
    class_label_map = {'animals': 0, 'animated': 1, 'barack_obama': 2, 'bernie_sanders': 3, 'bill_clinton': 4,
                       'donald_trump': 5, 'george_w_bush': 6, 'hillary_clinton': 7, 'joe_biden': 8, 'kamala_harris': 9,
                       'mike_pence': 10, 'nancy_pelosi': 11}

    saved_model_path = "../weights/model_effnet_politics_10epochs.h5"
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    outputs = []
    for filename in os.listdir(filepath):
        actual_class = filename.split('-')[0]
        test_image = Image.open(os.path.join(filepath, filename))
        test_image = np.array(test_image).astype('float32') / 255
        test_image = transform.resize(test_image, (480, 480, 3))
        test_image = np.expand_dims(test_image, axis=0)

        output = model.predict(test_image)
        arg_max_out = np.argmax(output[0])
        outputs.append(
            [filename, output, np.max(output), arg_max_out,
             list(class_label_map.keys())[list(class_label_map.values()).index(arg_max_out)], actual_class,
             class_label_map[actual_class], 'effnet_politics'])

    df_pred = pd.DataFrame(outputs,
                           columns=['file_name', 'pred_scores', 'pred_scores_max', 'pred_class_val', 'pred_class_',
                                    'actual_value', ' actual_class_val', 'model'])
    df_pred.to_csv('../outputs/effnet_politics_test.csv')
    return df_pred


# predict model using saved weights on meme dataset and predict file path  and returns dataframe of outputs
def predict_model_new_data(filepath, outfile='effnet_meme_politics_test.csv'):
    class_label_map = {'animals': 0, 'animated': 1, 'barack_obama': 2, 'bernie_sanders': 3, 'bill_clinton': 4,
                       'donald_trump': 5, 'george_w_bush': 6, 'hillary_clinton': 7, 'joe_biden': 8, 'kamala_harris': 9,
                       'mike_pence': 10, 'nancy_pelosi': 11}

    saved_model_path = "../weights/model_effnet_politics_10epochs.h5"
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    outputs = []
    for filename in os.listdir(filepath):
        test_image = Image.open(os.path.join(filepath, filename))
        test_image = np.array(test_image).astype('float32') / 255
        test_image = transform.resize(test_image, (480, 480, 3))
        test_image = np.expand_dims(test_image, axis=0)

        output = model.predict(test_image)
        arg_max_out = np.argmax(output[0])
        outputs.append(
            [filename, output, np.max(output), arg_max_out,
             list(class_label_map.keys())[list(class_label_map.values()).index(arg_max_out)], 'effnet_politics'])

    df_pred = pd.DataFrame(outputs,
                           columns=['file_name', 'pred_scores', 'pred_scores_max', 'pred_class_val', 'pred_class_',
                                    'model'])
    df_pred.to_csv('../outputs/' + outfile)
    return df_pred


# train and predict effnet model
def run_model(predict_path, outfile='effnet_meme_politics_test.csv', new_data=False):
    train_model()
    if new_data:
        return predict_model_new_data(predict_path, outfile)
    else:
        return predict_model(predict_path)


# run_model('../data/politics_test')
run_model('../data/MultiOFF/offensive', 'effnet_meme_politics_offensive_train.csv', True)
run_model('../data/MultiOFF/Non-offensive', 'effnet_meme_politics_nonoffensive_train.csv', True)
# run_model('../data/MultiOFF_Test', 'effnet_meme_politics_test.csv', True)
