"""
Author: Tan N. Le
Email: tnle@cs.stonybrook.edu
"""

from rating import Rating
import settings
import os
import pickle

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


    curr_data_points = 0;
    ratings = []
    movie_ids = []
    customer_ids = set()
    DEBUG=True

    def __init__(self, isRawData=True):
        self.log('init netflix data')

        if isRawData:
            for file_name in settings.data_files:
                self.load_data(file_name)
        else:
            self.loadFromFile()

    def load_data(self, file_name):
        if (self.curr_data_points >= settings.max_train_points & settings.max_train_points > 0):
            return

        self.log('Start reading file' + file_name + '\n')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #data_file = open(dir_path + '/' + settings.data_folder + '/' + file_name, 'r')
        stop = False
        movie_id = ''
        with open(dir_path + '/' + settings.data_folder + '/' + file_name) as data_file:
            for line in data_file:
            #line = data_file.readline()
                if not line:
                    stop = True
                    break

                if ':' in line:
                    movie_id = int(line.split(':')[0])
                    self.movie_ids.append(movie_id)

                else:
                    rating_data = line.split(',')
                    customer_id = int(rating_data[0])

                    self.customer_ids.add(customer_id)

                    rate_val = int(rating_data[1])
                    rating_date = rating_data[2].split('-')

                    rating_year = int(rating_date[0])
                    rating_month = int(rating_date[1])
                    rating_day = int(rating_date[2])

                    rating = Rating(movie_id, customer_id, rating_year, rating_month, rating_day, rate_val)
                    self.ratings.append(rating)
                    self.curr_data_points=self.curr_data_points+1

                if self.curr_data_points >= settings.max_train_points & settings.max_train_points > 0:
                    self.log(str(self.curr_data_points))
                    stop = True
                    break

        #data_file.close()

    def save2file(self):
        with open("super.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def loadFromFile(self):
        with open("super.file", "rb") as f:
            self = pickle.load(f)

    def log(self, str):
        if settings.DEBUG:
            print('[NetflixData] '+str)

