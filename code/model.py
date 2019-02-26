from keras.layers import Input, BatchNormalization, Conv1D, Activation, MaxPooling1D, Dropout, \
                            GlobalMaxPool1D, Dense
from keras.layers.merge import add
from keras.models import Model
import lightgbm as lgbm
from datautils import datagenerator
from keras.callbacks import TensorBoar

class ResNet:
    def __init__(self, inputshape, num_classes):
        self.model = self.resnet_with_dropout(inputshape, num_classes)

    def __resnet1D_block(self, num_filters, maxpool_size, inputlayer):
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

    def resnet_with_dropout(self, inputshape, num_classes):
        inputlayer = Input(shape=(inputshape[1],1))
        resnetblock = self.__resnet1D_block(64, 2, inputlayer)
        resnetblock = Dropout(0.3)(resnetblock)
        resnetblock = self.__resnet1D_block(128, 2, resnetblock)
        resnetblock = Dropout(0.3)(resnetblock)
        resnetblock = self.__resnet1D_block(256, 1, resnetblock)
        resnetblock = Dropout(0.3)(resnetblock)
        resnetblock = self.__resnet1D_block(512, 1, resnetblock)
        resnetblock = GlobalMaxPool1D()(resnetblock)
        resnetblock = Dropout(0.3)(resnetblock)
        resnetblock = Dense(4096, activation='relu')(resnetblock)
        resnetblock = Dense(2048, activation='relu')(resnetblock)
        resnetblock = Dropout(0.5)(resnetblock)
        softmax = Dense(num_classes, activation='sigmoid')(resnetblock)
        model = Model(inputlayer, softmax)
        return model

    def train(self, trainfile, validationfile, batchsize, num_steps, logdir):
        # refactor train function and args
        tensorboard = TensorBoar(logdir=logdir)
        self.model.compile(optimizer='adam', 
                            loss='binary_crossentropy', 
                            metrics=['accuracy', 'mse'])
        self.model.fit_generator(datagenerator(trainfile, batchsize),
                        steps_per_epoch=int(num_steps/1),
                        epochs=10, 
                        verbose=1,
                        validation_data=datagenerator(validationfile),
                        validation_steps=100,
                        callbacks = [tensorboard])
        return self.model


class LightGBM:
    def __init__(self, categorical_features):
        self.params = self.initparams()
        self.categorical_features = categorical_features

    def initparams(self):
        params = {
            'boosting_type': 'gbdt',
            'drop_rate': 0.09, 
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 33,  
            'max_depth': 5, 
            'min_child_samples': 100,
            'max_bin': 100,
            'subsample': 0.9,  
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'min_child_weight': 0,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'nthread': 4,
            'verbose': 0,
            'scale_pos_weight': 100 
        }
        return params
    
    def train(self, train_data, rounds):
        '''
        train_data: training dataset of type lightgbm.Dataset 
        rounds: number of training boositng rounds - you can use cross-validation to determine the optima boost_rounds
        '''
        model = lgbm.train(params=self.params, 
                            train_set=train_data, 
                            num_boost_round=rounds, 
                            categorical_feature=self.categorical_features)
        return model