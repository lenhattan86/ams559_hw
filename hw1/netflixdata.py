"""
Author: Tan N. Le
Email: tnle@cs.stonybrook.edu
"""

from rating import Rating
import settings
import os
import pickle

import tensorflow as tf

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
    movie_ids = []
    customer_ids = []
    DEBUG=True
    curr_offset=0

    def __init__(self, file_name, chunk_index):
        self.log('init netflix data')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_to_file = dir_path + '/' + settings.data_folder + '/' + file_name
        self.load_data_chunk(path_to_file, chunk_index)

    def load_data(self, file_name):
        if (self.curr_line >= settings.max_lines & settings.max_lines > 0):
            return

        self.log('Start reading file' + file_name + '\n')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        stop = False
        movie_id = ''
        with open(dir_path + '/' + settings.data_folder + '/' + file_name) as data_file:
            for line in data_file:
            #line = data_file.readline()
                if not line:
                    stop = True
                    break

                self.curr_line= self.curr_line + 1

                if ':' in line:
                    movie_id = int(line.split(':')[0])
                else:
                    self.movie_ids.append(movie_id)
                    rating_data = line.split(',')
                    customer_id = int(rating_data[0])

                    self.customer_ids.append(customer_id)

                    rate_val = int(rating_data[1])
                    rating_date = rating_data[2].split('-')

                    rating_year = int(rating_date[0])
                    rating_month = int(rating_date[1])
                    rating_day = int(rating_date[2])

                    self.ratings.append(rate_val)
                    self.years.append(rating_year)
                    self.days.append(rating_day)
                    self.months.append(rating_month)


                if self.curr_line >= settings.max_lines & settings.max_lines > 0:
                    self.log(str(self.curr_line))
                    stop = True
                    break

        data_file.close()

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
                    self.movie_ids.append(movie_id)
                    rating_data = line.split(',')
                    customer_id = int(rating_data[0])

                    self.customer_ids.append(customer_id)

                    rate_val = int(rating_data[1])
                    rating_date = rating_data[2].split('-')

                    rating_year = int(rating_date[0])
                    rating_month = int(rating_date[1])
                    rating_day = int(rating_date[2])

                    self.ratings.append(rate_val)
                    self.years.append(rating_year)
                    self.days.append(rating_day)
                    self.months.append(rating_month)

    def save2file(self):
        with open("super.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def loadFromFile(self):
        with open("super.file", "rb") as f:
            self = pickle.load(f)

    def log(self, str):
        if settings.DEBUG:
            print('[NetflixData] '+str)

    def save2_tfrecord(self):
        train_filename = 'train.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)
        for i in range(len(train_addrs)):

            # Load the image
            img = load_image(train_addrs[i])
            label = train_labels[i]
            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()
