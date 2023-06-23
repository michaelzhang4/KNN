import pandas as pd
import sklearn
from sklearn import preprocessing, neighbors

data = pd.read_csv("car.data")  # read in data

le = preprocessing.LabelEncoder()   # function that changes text into numerical values based on alphabetical order

buying = le.fit_transform(data["buying"])   # changes acc -> 0, good -> 1, unacc -> 2, vgood -> 3
maint = le.fit_transform(data["maint"])     # changes high -> 0, low -> 1, med -> 2, vhigh -> 3
doors = le.fit_transform(data["doors"])     # changes 2 -> 0, 3 -> 1, 4 -> 2, 5more -> 3
persons = le.fit_transform(data["persons"]) # changes 2 -> 0, 4 -> 1, more -> 2
lug_boot = le.fit_transform(data["lug_boot"])   # changes big -> 0, med -> 1, small -> 2
safety = le.fit_transform(data["safety"])   # changes high -> 0, low -> 1, med -> 2
cls = le.fit_transform(data["class"])   # changes acc -> 0, good -> 1, unacc -> 2, vgood -> 3

X = list(zip(buying,maint,lug_boot,doors,safety,persons)) # values of each dimension of coordinate points

y = list(cls)   # the class of each entry, which is the predicted value

model = neighbors.KNeighborsClassifier(n_neighbors=9)   # KNN model with k = 9 as there is enough entries to compare with the 9 nearest points

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,train_size=0.9) # splits the data into training and test sets

model.fit(x_train,y_train)  #   trains the model

names = ["acc","good","unacc","vgood"]  # for referring numbers back to original classes, preprocessing orders them alphabetically

predictions = model.predict(x_test) # predict class of each entry based on coordinates

for x in range(len(predictions)):   # print the predicted class, the coordinate values of the entry and the actual class
    print(names[predictions[x]], x_test[x], names[y_test[x]])

print(model.score(x_test,y_test))   # rates the accuracy of the model on percentage of correct predictions