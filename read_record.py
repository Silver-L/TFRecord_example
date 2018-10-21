import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    file_name = ['./tfrecord/recordfile_{}'.format(i+1) for i in range(60)]

    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(lambda x: _parse_function(x, image_size=[28, 28]), num_parallel_calls=os.cpu_count())
    dataset = dataset.shuffle(buffer_size = 10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(256)
    iterator = dataset.make_one_shot_iterator()
    X = iterator.get_next()

    with tf.Session() as sess:
        mnist = sess.run(X)

    plt.figure(figsize=(10, 10))
    plt.imshow(mnist[1], origin="upper", cmap="gray")
    plt.show()

# # load tfrecord function
def _parse_function(record, image_size=[28, 28, 1]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image

if __name__ == '__main__':
    main()