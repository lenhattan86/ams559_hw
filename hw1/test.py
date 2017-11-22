import numpy
import pandas as pd


data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
pd.DataFrame(data=data[1:,1:],    # values
           index=data[1:,0],    # 1st column as index
              columns=data[0,1:])

include_index = [1,2,3]
include_idx = set(include_index)  #Set is more efficient, but doesn't reorder your elements if that is desireable
mask = numpy.array([(i in include_idx) for i in xrange(len(a))])

included = a[mask]  # array([0, 1, 2, 3])
excluded = a[~mask] # array([4, 5, 6, 7, 8, 9])

print included