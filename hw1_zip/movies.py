import settings
import os

class Movies:
    movie_ids=set()
    movie_years=[]
    movie_titles=[]
    def __init__(self):
        self.log('init Movies')
        self.load_data(settings.movie_file)

    def load_data(self, file_name):
        self.log('Start reading file' + file_name + '\n')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        stop = False
        with open(dir_path + '/' + settings.data_folder + '/' + file_name) as data_file:
            for line in data_file:

                if not line:
                    stop = True
                    break

                #print(line)



                movie_data = line.split(',')
                movie_id = int(movie_data[0])

                if 'NULL' in movie_data[1]:
                    movie_data[1]='-1'

                movie_year=int(movie_data[1])

                self.movie_ids.add(movie_id)
                self.movie_years.append(movie_year)

        data_file.close()

    def log(self, str):
        if settings.DEBUG:
            print('[Movies] '+str)
