from netflixdata import NetflixData
import settings
from movies import Movies
import time


def log(str):
    print('[Main] '+str)

def main():

    # parameters
    isSavingData = False

    #settings.init()
    print '=========== NETFLIX RATING PREDICTION =========='

    movies = Movies()
    if settings.IS_TEST_DATA:
        log('Num of movies '+ str(len(movies.movie_ids)))

    print '=========== DATA LOADING... =========='
    start_time = time.time()

    nexflix_data = NetflixData(settings.data_files[0], 4)

    if settings.IS_TEST_DATA:
        log('Num of data points '+ str(len(nexflix_data.ratings)))

    end_time = time.time()
    log('loading data takes ' + str(end_time - start_time) + ' seconds')

    if isSavingData:
        print '=========== SAVING DATA... =========='
        start_time = time.time()
        nexflix_data.save2file()
        end_time = time.time()
        log('saving data takes ' + str(end_time - start_time) + ' seconds')


    print '============ TRAINING ==========='

    print '=========== EVALUATING =========='

if __name__ == "__main__": main()
