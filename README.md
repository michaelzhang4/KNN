## About ##

This is a K-nearest-neighbours model on a car dataset without raw data in numeric-form.
I process the data, changing into numeric form then split the data into training and test sets.
I used the training sets to fit the KNN-model and the test sets to gauge the accuracy of the final model.
I did not create a pickle file to save the model as with KNN it is useless due
to the fact that for every new point entered, the model has to recompute the distance
between the new entry and every other point to know how to classify it.