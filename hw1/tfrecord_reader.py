import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = 'gen/train.tfrecord'  # address to save the hdf5 file

with tf.Session() as sess:
    feature = {'train/user': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)   

    # Cast label data into int32
    label = tf.cast(features['train/user'], tf.int32)
    
    # Creates batches by randomly shuffling tensors
    labels = tf.train.shuffle_batch([label], batch_size=1, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        lbl = sess.run([labels])
        print lbl
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()