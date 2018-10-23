import os
import tensorflow as tf

def main():

    outdir = './tfrecord'

    # check folder
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)

    mnist = tf.keras.datasets.mnist
    (data_set, _ ), ( _ , _ ) = mnist.load_data()

    num_per_tfrecord = 1000
    num_of_total_image = data_set.shape[0]

    if (num_of_total_image % num_per_tfrecord != 0):
        num_of_recordfile = num_of_total_image // num_per_tfrecord + 1
    else:
        num_of_recordfile = num_of_total_image // num_per_tfrecord
        
    num_per_tfrecord_final = num_of_total_image - num_per_tfrecord * (num_of_recordfile - 1)

    print('number of total TFrecordfile: {}'.format(num_of_recordfile))

    # write TFrecord
    for i in range(num_of_recordfile):
        tfrecord_filename = os.path.join(outdir, 'recordfile_{}'.format(i + 1))
        write = tf.python_io.TFRecordWriter(tfrecord_filename)

        print('Writing recordfile_{}'.format(i+1))
        
        if i == num_of_recordfile - 1:
            loop_buf = num_per_tfrecord_final
        else :
            loop_buf = num_per_tfrecord

        for image_index in range(loop_buf):
            image = data_set[image_index + i*num_per_tfrecord].flatten()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                }))

            write.write(example.SerializeToString())
        write.close()

if __name__ == '__main__':
    main()
