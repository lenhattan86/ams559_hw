import math

num_chunks_a_file=1;
##################################
USE_PROBE = True
LOG_FILE='log'
TRAIN_RATIO = 0.8
BATCH_SIZE = 10000
EPOCH_MAX = 1
DEVICE = "/cpu:0"
learning_rate = 0.01
REGULARIZATION = 10 # 0.0 0.05 0.1 0.15 0.2 0.25
###############################

##
DEBUG = False

#
IS_TEST_DATA = True

##
#global customersId
#movie_ids=range()

# 100480507


##
data_folder='netflix-prize-data'
data_files=['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
# data_files=['combined_data_1.txt']

#data_folder='sample'
#data_files=['data_sample.txt', 'data_sample.txt']

qualify_file=['qualifiying.txt']
probe_file='probe.txt'

movie_file='movie_titles.csv'

## Linear Regression

#
NUM_THREADS=8

#
NUM_USERS=2649429
#
NUM_MOVIES=17770

USER_NUM = 4
MOVIE_NUM = 5
DIM = 3


DATA_LEN=15



TRAIN_FILE='gen/train.tfrecord'
VAL_FILE='gen/val.tfrecord'
TEST_FILE='gen/test.tfrecord'