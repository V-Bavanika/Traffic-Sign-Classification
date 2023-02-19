#Step 1 : Explore the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from sklearn.metrics import accuracy_score

data = []
labels = []
classes = 43
#OS - Py standard utility modules
#getting the path of current working directory
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


#here we r stacking the different con. layers.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))#activation func = relu
model.add(MaxPool2D(pool_size=(2,2)))# generally Convolutional Layer is followed by a Pooling Layer.(max element is choosen)
model.add(Dropout(0.25))#done to prevent overfitting
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))#256 neurons
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))#last layer has softmax for multi class classiication  prob distib.

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#Adaptive Moment Estimation use :Momentum and Adaptive Learning Rates to #converge faster. 

epochs = 15
history = model.fit(X_train, y_train, epochs=epochs,batch_size=32, validation_data=(X_test, y_test))
model.save('traffic_signs.h5') # .h5 choosen because of Keras lib. & too many numeric values to store.

plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.show()

#Testing accuracy of test data
y_test = pd.read_csv('Test.csv')

labels = y_test['ClassId'].values
imgs = y_test['Path'].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test = np.array(data)

predict_x = model.predict(X_test)
classes_x = np.argmax(predict_x, axis=1)

print("Accuracy of test data : ",accuracy_score(labels, classes_x))

model.save('traffic_classifier.h5')