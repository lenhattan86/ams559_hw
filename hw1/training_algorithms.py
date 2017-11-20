import settings

import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

import dataio
import ops
import os

def clip(x):
    return np.clip(x, 1.0, 5.0)

def get_data():
    df = dataio.read_process("/tmp/movielens/ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * settings.TEST_RATIO)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def svd():

    ## build the model
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    movie_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rating_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, movie_batch, user_num=settings.USER_NUM+1, item_num=settings.ITEM_NUM+1, dim=settings.DIM,
                                           device=settings.DEVICE)

    global_step = tf.train.get_or_create_global_step()

    _, train_op = ops.optimization(infer, regularizer, rating_batch, learning_rate=0.001, reg=0.05, device=settings.DEVICE)

    init_op = tf.global_variables_initializer() ##??

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    summary_file = open(curr_dir+'/'+settings.LOG_FILE, 'w')
    print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))        
    errors = deque(maxlen=settings.BATCH_SIZE)

    with tf.Session() as sess:

        feature = {'train/user': tf.FixedLenFeature([], tf.int64),
                'train/movie': tf.FixedLenFeature([], tf.int64),
                'train/rating': tf.FixedLenFeature([], tf.int64)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([settings.TRAIN_FILE], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)   

        # Cast user data into int32
        user = tf.cast(features['train/user'], tf.int32)
        movie = tf.cast(features['train/movie'], tf.int32)
        rating = tf.cast(features['train/rating'], tf.int32)

        # Creates batches by randomly shuffling tensors
        user_queue, movie_queue, rating_queue = tf.train.shuffle_batch([user, movie, rating], batch_size=settings.BATCH_SIZE, \
            capacity=settings.BATCH_SIZE*30, num_threads=1, min_after_dequeue=settings.BATCH_SIZE*10)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        ############ 
        #sess.run(init_op)
        start = time.time()

        ##########
        for i in range(1):

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            

            for iBatch in range(settings.DATA_LEN / settings.BATCH_SIZE):
                users, movies, ratings = sess.run([user_queue, movie_queue, rating_queue]) 

                _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                       movie_batch: movies,
                                                                       rating_batch: ratings}) 
                pred_batch = clip(pred_batch)
                errors.append(np.power(pred_batch - ratings, 2))  

            train_err = np.sqrt(np.mean(errors))                                
            end = time.time()
            test_err = -1
            print("{:3d} {:f} {:f} {:f}(s)".format(i , train_err, test_err,
                                                   end - start))
            summary_file.write("{:3d},{:f},{:f},{:f}\n".format(i , train_err, test_err,
                                                   end - start))
            start = end
            # Stop the threads
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)

        sess.close()
        summary_file.close()