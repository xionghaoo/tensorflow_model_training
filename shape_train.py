import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

print(tf.__version__)

dsprites_train, ds_info = tfds.load(
    name="dsprites",
    split="train",
    shuffle_files=True,
    with_info=True
)

print(ds_info)

def normalize_img(item):
    label = item['label_shape']
    print(label)
    return tf.cast(item['image'], tf.float32) / 255., label


dsprites_train = dsprites_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dsprites_train = dsprites_train.cache()
dsprites_train = dsprites_train.shuffle(int(ds_info.splits['train'].num_examples / 100))
dsprites_train = dsprites_train.batch(128)
dsprites_train = dsprites_train.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# print(dsprites_train)

model.fit(dsprites_train, epochs=10)
