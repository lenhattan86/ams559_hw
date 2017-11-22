from netflixdata import NetflixData
import settings
from movies import Movies
import time
import training_algorithms as alg


def log(str):
    print('[Main] '+str)

def main():

    # parameters
    isSavingData = True

    #settings.init()
    print '=========== NETFLIX RATING PREDICTION =========='

    #movies = Movies()
    #if settings.IS_TEST_DATA:
    #    log('Num of movies '+ str(len(movies.movie_ids)))

    print '=========== DATA loading... =========='
    start_time = time.time()

    settings.DATA_LEN=0
    nexflix_data = NetflixData()

    settings.LOG_FILE = settings.LOG_FILE + '_' + str(settings.EPOCH_MAX) + '_' + str(settings.REGULARIZATION) + '_' + str(settings.learning_rate) + '.csv'

    settings.USER_NUM = nexflix_data.USER_NUM
    settings.MOVIE_NUM = nexflix_data.MOVIE_NUM
    
    log('Num of data points '+ str(settings.DATA_LEN))
    log('Num of users '+ str(nexflix_data.USER_NUM))
    log('Num of movies '+ str(nexflix_data.MOVIE_NUM))
    # log('Num of test data points '+ str(len(nexflix_data.test_ratings)))

    end_time = time.time()
    log('loading data takes ' + str(end_time - start_time) + ' seconds')

    print '============ TRAINING ==========='
    start_time = time.time()
    alg.svd(nexflix_data.df_train, nexflix_data.df_test)
    end_time = time.time()
    log('Training takes ' + str(end_time - start_time) + ' seconds')

    log(settings.LOG_FILE)

    print '============= PREDICTING =========='

    print '=========== EVALUATING =========='

if __name__ == "__main__": main()
