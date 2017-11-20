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

def svd(train, test):
    samples_per_batch = len(train)

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=settings.BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    movie_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rating_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, movie_batch, user_num=settings.USER_NUM, item_num=settings.ITEM_NUM, dim=settings.DIM,
                                           device=settings.DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rating_batch, learning_rate=0.001, reg=0.05, device=settings.DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   movie_batch: items,
                                                                   rating_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            movie_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
def svd():
    samples_per_batch = 2 #len(train)

    ## build the model
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    movie_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rating_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, movie_batch, user_num=settings.USER_NUM+1, item_num=settings.ITEM_NUM+1, dim=settings.DIM,
                                           device=settings.DEVICE)

    global_step = tf.train.get_or_create_global_step()

    _, train_op = ops.optimization(infer, regularizer, rating_batch, learning_rate=0.001, reg=0.05, device=settings.DEVICE)

    init_op = tf.global_variables_initializer() ##??

    ######################

    with tf.Session() as sess:
        feature = {'train/user': tf.FixedLenFeature([], tf.int64),
                    'train/movie': tf.FixedLenFeature([], tf.int64),
                    'train/rating': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([settings.TRAIN_FILE, settings.TRAIN_FILE], num_epochs=1)
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
        user_queue, movie_queue, rating_queue = tf.train.shuffle_batch([user, movie, rating], batch_size=2, capacity=30, num_threads=1, min_after_dequeue=10)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        ############ 
        #sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()

        ##########

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #for batch_index in range(5):
        #    train_ratings, train_users, train_movies = sess.run([ratings, users, movies])

        #for i in range(settings.EPOCH_MAX * samples_per_batch):
        for i in range(6 * samples_per_batch):
            users, movies, ratings = sess.run([user_queue, movie_queue, rating_queue])                    

            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   movie_batch: movies,
                                                                   rating_batch: ratings})            
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - ratings, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))                                
                end = time.time()
                test_err = -1
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

def svd_bk():
    samples_per_batch = 15 #len(train)

    ## build the model

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    movie_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rating_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, movie_batch, user_num=settings.USER_NUM, item_num=settings.ITEM_NUM, dim=settings.DIM,
                                           device=settings.DEVICE)

    #global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.train.get_or_create_global_step()

    _, train_op = ops.optimization(infer, regularizer, rating_batch, learning_rate=0.001, reg=0.05, device=settings.DEVICE)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        ## load tf record 
        feature = {'train/user': tf.FixedLenFeature([], tf.int64),
                   'train/movie': tf.FixedLenFeature([], tf.int64),
                   'train/rating': tf.FixedLenFeature([], tf.int64)}

        dir_path = os.path.dirname(os.path.realpath(__file__))

        # 1 & 2. Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([dir_path+ '/' + settings.TRAIN_FILE], num_epochs=1)

        # 3. Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # 4. Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # 5. Cast data into int32
        user = tf.cast(features['train/user'], tf.int32)
        movie = tf.cast(features['train/movie'], tf.int32)
        rating = tf.cast(features['train/rating'], tf.int32)

        # Any preprocessing here ...
        print 'user type ' + str(user)

        # Creates batches by randomly shuffling tensors
        ratings = tf.train.shuffle_batch([rating], \
            batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)



        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(settings.EPOCH_MAX * samples_per_batch):

            #users, items, rates = next(iter_train)
            rating = sess.run([ratings])

            #print rating

            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   movie_batch: items,
                                                                   rating_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            movie_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end                
