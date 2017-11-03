

class Rating:
    """
    this class describles the rating. It has the following properties
    customer_id

    movie_id

    year
    month
    day
    rating
    """


    def __init__(self, customer_id, movie_id, year, month, day, rating):
        self.customer_id = customer_id

        self.movie_id = movie_id

        self.year = year
        self.month = month
        self.day = day

        self.rating = rating
