from tensorflow.keras.datasets import fashion_mnists
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

# fashion mist returns 2 pair of values.
(x_train,y_train),(x_test,y_test) = fashion_mnists.load_data()

#Use smaller dataset for knn
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

#Convert the 28x28 into 1D
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)
# -1 —> automatically calculates the remaining dimension.

#Normalize it
x_train = x_train / 255
x_test = x_test / 255

for k in [1,3,5,7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,pred)
    print("K = ",k,"Accuracy = ",accuracy)
