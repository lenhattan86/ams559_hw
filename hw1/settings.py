import math

##
DEBUG = False

##
max_lines = 50000000;

#
max_buffer_to_load=math.pow(2,2*10)

#
num_chunks_a_file=1;

#
IS_TEST_DATA = True

##
#global customersId
#movie_ids=range()


##
#data_files=['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
#data_files=['combined_data_1.txt']
data_files=['data_sample.txt']
data_folder='netflix-prize-data'

movie_file='movie_titles.csv'

## Linear Regression

training_epochs=1000
learning_rate = 0.01

## device
device="/cpu:0"
#device="/gpu:0"

#
NUM_THREADS=8

#
NUM_USERS=4
#
NUM_MOVIES=5
#
NUM_FEATURES=3
