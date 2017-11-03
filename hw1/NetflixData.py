

class NetflixData(object):
    """
    This class loads the Netflixdata and privides with some util functions. NetflixData has the following properties

    Attributes:
        data_folder
        data_files

        num_movies
        num_users

        probe_file

        qualifying_file
    """
#    data_files=['combined_data_1.txt','combined_data_2.txt']
    data_folder="./netflix-prize-data"

    def __init__(self):
		super(self).__init__()
        self.data_folder="./netflix-prize-data"
        self.data_files=['combined_data_1.txt','combined_data_2.txt']
        for file_name in self.data_files:
            self.load_data(file_name)

    def load_data(self, file_name):
        file_data = open(self.data_folder + '/' + file_name, 'r')
