import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train, test = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train, y_train = train
x_test, y_test = test

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Model
randomForest = RandomForestClassifier()
randomForest.fit(x_train, y_train)

predictions = randomForest.predict(x_test)
print(classification_report(y_test, predictions))