from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cross_validation import train_test_split

def get_model():
    try:
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            model = load_model('my_model.h5')
    except:
        print("You must run create_model.py before training")
        sys.exit()
    return model

def get_data():
    try:
        X=np.load("X.npy")
        Y=np.load("Y.npy")
    except:
        print("Run preprocess.py")
    return train_test_split(X,Y)


def train(model,X_train,Y_train,xt,yt,numb_epochs):
    datagen=ImageDataGenerator(rotation_range=180,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,fill_mode='nearest')
    es=EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4,verbose=1, mode='auto')
    history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=32),steps_per_epoch=200,validation_data=(xt,yt),epochs=numb_epochs,callbacks=[es])
    return history

def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("1.jpg")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("2.jpg")

def main():
    data=get_data()
    model=get_model()
    hist=train(model,epochs,data[0],data[2],data[1],data[3],5)
    plot(hist)

if __name__ == '__main__':
    main()
