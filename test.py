import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

dsprites_train, ds_info = tfds.load(
    name="dsprites",
    split="train",
    with_info=True
)
assert isinstance(dsprites_train, tf.data.Dataset)

dsprites_train = dsprites_train.prefetch(tf.data.experimental.AUTOTUNE)
print(dsprites_train)

images = []
labels = []
for example in dsprites_train.take(3):
    image, label = example["image"], example["label_shape"]
    images.append(image.numpy()[:, :, 0].astype(np.float32))
    labels.append(label.numpy())
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()

print("images size: %d", len(images))
print("labels size: %d", len(labels))

# images = images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(images)
print(labels)


model.fit(images, epochs=10)

# def normalize_img(train):
#     print(train["image"])
#     return tf.cast(train["image"], tf.float32) / 255.
#
#
# dsprites_train = dsprites_train.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dsprites_train = dsprites_train.cache()
# dsprites_train = dsprites_train.shuffle(ds_info.splits['train'].num_examples)
# dsprites_train = dsprites_train.batch(128)
# dsprites_train = dsprites_train.prefetch(tf.data.experimental.AUTOTUNE)
#
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(64, 64)),
#     tf.keras.layers.Dense(128,activation='relu'),
#     tf.keras.layers.Dense(10)
# ])
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )
#
# print(ds_info.features)
#
# print(dsprites_train)

# model.fit(
#     dsprites_train,
#     epochs=10,
#     # validation_data=ds_test
# )



