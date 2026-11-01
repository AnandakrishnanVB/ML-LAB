from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

#Convert into 1D
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

#normalize it
x_train = x_train/255
x_test = x_test/255

for k in [1,3,5,7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,pred)
    print("K:",k,"accuracy:",accuracy)

