# First data drift observations

The Iris training data has four input measurements: sepal length, sepal width, petal length, and petal width. Any of them could change in production if the flower population, measurement process, or sensor calibration changes.

A shift in petal length or width is especially important because the model uses those measurements to separate species. A changed distribution could move observations across learned decision boundaries, causing more incorrect predictions and lower accuracy.

`src/drift.py` prints the reference dataset's count, mean, standard deviation, quartiles, and range. To check real drift later, compare those statistics with a fresh production sample. This TP has no production sample, so no actual drift has been established.
