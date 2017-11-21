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

    print '=========== DATA Converting... =========='
    start_time = time.time()

    settings.DATA_LEN=0
    nexflix_data = NetflixData()
    
    log('Num of data points '+ str(settings.DATA_LEN))

    end_time = time.time()
    log('loading data takes ' + str(end_time - start_time) + ' seconds')

    print '============ TRAINING ==========='
    start_time = time.time()
    #alg.svd()
    end_time = time.time()
    log('Training takes ' + str(end_time - start_time) + ' seconds')

    print '============= PREDICTING =========='

    print '=========== EVALUATING =========='

if __name__ == "__main__": main()
