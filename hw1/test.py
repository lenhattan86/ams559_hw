import numpy
import pandas as pd
import array


numpy.random.seed(123)
e = numpy.random.normal(size=10)  
dataframe=pd.DataFrame(e, columns=['a']) 

include_index = [1,2,3]
include_idx = set(include_index)  #Set is more efficient, but doesn't reorder your elements if that is desireable
mask = numpy.array([(i in include_idx) for i in xrange(len(dataframe))])

included = dataframe[mask]  # array([0, 1, 2, 3])
excluded = dataframe[~mask] # array([4, 5, 6, 7, 8, 9])

print included
print excluded