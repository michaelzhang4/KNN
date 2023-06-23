## About ##

This is a K-nearest-neighbours model on a car dataset. The raw data is not in numeric-form so the data is processed, changing it into numeric form before splitting the data into training and test sets.
The training set is used to fit the KNN-model and the test set to gauge the accuracy of the final model.
A pickle file to save the model was not included as with KNN it is useless due
to the fact that for every new point entered, the model has to recompute the distance
between the new entry and every other point to know how to classify it.
