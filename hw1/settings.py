import math

##
global DEBUG
DEBUG = False

##
global max_lines
max_lines = 50000000;


global max_buffer_to_load
max_buffer_to_load=math.pow(2,2*10)


global num_chunks_a_file
num_chunks_a_file=1000;

##
global IS_TEST_DATA
IS_TEST_DATA = True

##
#global customersId
#movie_ids=range()


##
#data_files=['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
data_files=['combined_data_1.txt']
#data_files=['sample.txt']
data_folder='netflix-prize-data'

movie_file='movie_titles.csv'
