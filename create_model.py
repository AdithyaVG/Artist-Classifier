from keras.applications import mobilenet
from keras.layers import Dense, Activation,AveragePooling2D
from keras import regularizers
from keras import Model

def pre_trainedmodel():
    model=mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224,224,3))
    X=AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid', data_format=None)(model.output)
    X=Dense(209,activation='softmax',input_shape=(1,1,1024),kernel_regularizer=regularizers.l2(0.01))(X)
    for layer in model2.layers[:-1]:
      layer.trainable=False
    model=Model(model.input,X)
    return model

def main():
    model=pre_trainedmodel()
    model.save('my_model.h5')

if __name__ == '__main__':
    main()
