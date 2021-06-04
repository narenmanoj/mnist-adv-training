import tensorflow as tf


def convert_to_tfds(examples, labels, batch_size=32, buffer_size=100):
    if buffer_size == 0:
        tf.data.Dataset.from_tensor_slices(
            (examples, labels)).batch(batch_size)
    return tf.data.Dataset.from_tensor_slices(
        (examples, labels)).shuffle(buffer_size).batch(batch_size)
