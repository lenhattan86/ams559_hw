"""
Author: Tan N. Le
Email: tnle@cs.stonybrook.edu
"""

from rating import Rating
import settings
import os
import pickle

import tensorflow as tf
import utils
import sys

class NetflixData(object):
    """
    This class loads the Netflixdata and privides with some util functions. NetflixData has the following properties

    Attributes:
        data_folder
        data_files

        movie_ids
        customer_ids

        ratings
    """

    binary_data='rating.dat'

    curr_line = 0;
    ratings = []
    years = []
    months =  []
    days = []
    movies = []
    users = []
    DEBUG=True
    curr_offset=0

    def __init__(self, file_name, chunk_index):
        self.log('init netflix data')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_file = dir_path + '/' + settings.data_folder + '/' + file_name
        self.load_data_chunk(path_to_file, chunk_index)

    def load_data_chunk(self, path_to_file, chunk_index):
        assert 0 <= chunk_index and chunk_index < settings.num_chunks_a_file

        movie_id=''
        with open(path_to_file) as data_file:
            data_file.seek(0,2)
            file_size = data_file.tell()

            ini = file_size * chunk_index / settings.num_chunks_a_file
            end = file_size * (1 + chunk_index) / settings.num_chunks_a_file

            if ini <= 0:
                data_file.seek(0)
            else:
                data_file.seek(ini-1)
                # is readline necessary?
                data_file.readline()

            while data_file.tell() < end:
                line = data_file.readline()

                if ':' in line:
                        movie_id = int(line.split(':')[0])
                else:
                    self.movies.append(movie_id)
                    rating_data = line.split(',')
                    customer_id = int(rating_data[0])

                    self.users.append(customer_id)

                    rate_val = int(rating_data[1])
                    rating_date = rating_data[2].split('-')

                    rating_year = int(rating_date[0])
                    rating_month = int(rating_date[1])
                    rating_day = int(rating_date[2])

                    self.ratings.append(rate_val)
                    #self.years.append(rating_year)
                    #self.days.append(rating_day)
                    #self.months.append(rating_month)
        ## todo: shuffle the data.

        

    def save2file(self):
        with open("super.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def loadFromFile(self):
        with open("super.file", "rb") as f:
            self = pickle.load(f)

    def log(self, str):
        if settings.DEBUG:
            print('[NetflixData] '+str)

    def save_all_tf_record(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ## transfer to training, validation, and errors
        train_idx = int(settings.TRAIN_RATIO*len(self.ratings))
        train_users = self.users[0:train_idx]
        train_movies = self.movies[0:train_idx]
        train_ratings = self.ratings[0:train_idx]
#        print train_users
#        print train_movies
#        print train_ratings
        self.save_tfrecord(dir_path + '/' + settings.TRAIN_FILE, train_users, train_movies, train_ratings)

    #    val_idx = int(settings.VAL_RATIO*len(self.ratings))
    #    val_users = self.users[train_idx:val_idx]
    #    val_movies =  self.movies[train_idx:val_idx]
    #    val_ratings = self.ratings[train_idx:val_idx]
    #    self.save_tfrecord(dir_path + '/' + settings.VAL_FILE, val_users, val_movies, val_ratings)

    #    test_idx = int(settings.TEST_RATIO*len(self.ratings))
    #    test_users = self.users[val_idx:test_idx]
    #    test_movies = self.users[val_idx:test_idx]
    #    test_ratings = self.users[val_idx:test_idx]
    #    self.save_tfrecord(dir_path + '/' + settings.TEST_FILE, test_users, test_movies, test_ratings)


    def save_tfrecord(self, path_to_file, users, movies, ratings):
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(path_to_file)
        for i in range(len(users)):
            user = users[i]
            movie = movies[i]
            rating = ratings[i]
            # Create a feature
            feature = {'train/user': utils._int64_feature(user),
                       'train/movie': utils._int64_feature(movie),
                       'train/rating': utils._int64_feature(rating)}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()
