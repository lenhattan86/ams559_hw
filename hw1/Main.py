from netflixdata import NetflixData
import settings
import time


def log(str):
    print('[Main] '+str)

def main():

    # parameters
    isRawData = True
    isSavingData = False

    #settings.init()
    print '=========== NETFLIX RATING PREDICTION =========='


    print '=========== DATA LOADING... =========='
    start_time = time.time()
    nexflix_data = NetflixData(isRawData)
    if settings.IS_TEST_DATA:
        log('Num of data points '+ str(len(nexflix_data.ratings)))
        log('Num of movies '+ str(len(nexflix_data.movie_ids)))
        log('Num of customers '+ str(len(nexflix_data.customer_ids)))
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


