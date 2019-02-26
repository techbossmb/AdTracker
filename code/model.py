from keras.layers import Input, BatchNormalization, Conv1D, Activation, MaxPooling1D, Dropout, \
                            GlobalMaxPool1D, Dense
from keras.layers.merge import add
from keras.models import Model

def __resnet1D_block(num_filters, maxpool_size, inputlayer):
    skipblock = Conv1D(num_filters, 1, padding='same')(inputlayer)
    convblock = Conv1D(num_filters, 3, padding='same')(convblock)
    convblock = BatchNormalization()(convblock)
    convblock = Activation('relu')(convblock)
    convblock = Conv1D(num_filters, 3, padding='same')(convblock)
    convblock = add([convblock, skipblock])
    convblock = BatchNormalization()(convblock)
    convblock = Activation('relu')(convblock)
    convblock = MaxPooling1D(maxpool_size)(convblock)
    return convblock



def resnet_with_dropout(inputshape, num_classes):
	inputlayer = Input(shape=(inputshape[1],1))
    resnetblock = __resnet1D_block(64, 2, inputlayer)
    resnetblock = Dropout(0.3)(resnetblock)
    resnetblock = __resnet1D_block(128, 2, resnetblock)
    resnetblock = Dropout(0.3)(resnetblock)
    resnetblock = __resnet1D_block(256, 1, resnetblock)
    resnetblock = Dropout(0.3)(resnetblock)
    resnetblock = __resnet1D_block(512, 1, resnetblock)
    resnetblock = GlobalMaxPool1D()(resnetblock)
    resnetblock = Dropout(0.3)(resnetblock)
    resnetblock = Dense(4096, activation='relu')(resnetblock)
    resnetblock = Dense(2048, activation='relu')(resnetblock)
    resnetblock = Dropout(0.5)(resnetblock)
    softmax = Dense(num_classes, activation='sigmoid')(resnetblock)
    model = Model(inputlayer, softmax)
    return model