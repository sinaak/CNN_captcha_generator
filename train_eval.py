import numpy as np
np.random.seed(1337)  # for reproducibility

import building_data
import random
import os
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import h5py
from keras.models import model_from_json

batch_size = 128
nb_epoch = 64

MAX_CAPTCHA = building_data.get_captcha_size()
CHAR_SET_LEN = building_data.get_char_set_len()
Y_LEN = building_data.get_y_len()


# input image dimensions
img_rows, img_cols = building_data.get_height(), building_data.get_width()




def load_data(tol_num,train_num):
      
    # input,tol_num: the numbers of all samples(train and test)
    # input,train_num: the numbers of training samples
    # output,(X_train,y_train):trainging data
    # ouput,(X_test,y_test):test data
    
    data = np.empty((tol_num, 1, img_rows, img_cols),dtype="float32")
    label = np.empty((tol_num,Y_LEN),dtype="uint8")

    # data dir
    imgs = os.listdir("data")
    
    for i in range(tol_num):
        # load the images and convert them into gray images
        img = Image.open("data/"+imgs[i])

        arr = np.asarray(img,dtype="float32")
        try:
            data[i,:,:,:] = arr
            captcha_text = imgs[i].split('.')[0].split('_')[1]

            text_len = len(captcha_text)
            if text_len > MAX_CAPTCHA:
                raise ValueError(MAX_CAPTCHA)
                # the shape of the vector is 1*(MAX_CAPTCHA*CHAR_SET_LEN)
            vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
            def char2pos(c):
                k = CHAR_SET.index(c)
                return k
            for i, c in enumerate(captcha_text):
                idx = i * CHAR_SET_LEN + char2pos(c)
                vector[idx] = 1

            label[i] =vector

        except:
            pass

    # the data, shuffled and split between train and test sets
    rr = [i for i in range(tol_num)] 
    random.shuffle(rr)
    X_train = data[rr][:train_num]
    y_train = label[rr][:train_num]
    X_test = data[rr][train_num:]
    y_test = label[rr][train_num:]
    
    return (X_train,y_train),(X_test,y_test)


# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = load_data(tol_num = 20000,train_num = 18000)



# i use the theano backend
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


def show_history(history,label):
    # list all data in history
    import matplotlib.pyplot as plt
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.savefig(label+'accuracy'+'.png', bbox_inches='tight')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.savefig(label+'loss'+'.png', bbox_inches='tight')
    plt.close()
    



model = Sequential()


# 3 conv layer
model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(MAX_CAPTCHA*CHAR_SET_LEN))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


hist_results = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(X_test,Y_test))


show_history(hist_results,'1_label')


score = model.evaluate(X_test, Y_test, verbose=0)
predict = model.predict(X_test,batch_size = batch_size,verbose = 0)

# calculate the accuracy with the test data
acc = 0

def get_max(array):
    max_num = max(array)
    for i in range(len(array)):
        if array[i] == max_num:
            return i

for i in range(X_test.shape[0]):
    true = []
    predict2 = []
    for j in range(MAX_CAPTCHA):
        true.append(get_max(Y_test[i,CHAR_SET_LEN*j:(j+1)*CHAR_SET_LEN]))
        predict2.append(get_max(predict[i,CHAR_SET_LEN*j:(j+1)*CHAR_SET_LEN]))
    if true == predict2:
        acc+=1
    if i<20:
        print (i,' true: ',true)
        print (i,' predict: ',predict2)
print('predict correctly: ',acc)
print('total prediction: ',X_test.shape[0])
print('Score: ',score)

# save model
json_string = model.to_json()
open("my_model.json","w").write(json_string)
model.save_weights('my_model_weights.h5')
