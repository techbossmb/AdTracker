from keras.layers import Input, BatchNormalization, Conv1D, Activation, MaxPooling1D, Dropout, \
                            GlobalMaxPool1D, Dense
from keras.layers.merge import add
from keras.models import Model
import lightgbm as lgbm
from datautils import datagenerator, load_dataset
from keras.callbacks import TensorBoar
import os

class ResNet:
    def __init__(self):
        self.params = self.__init_params()

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

    def __init_params(self):
        params = {
            'num_features': 5,
            'num_classes': 1,
            'batchsize': 1280,
            'num_steps': 1000,
            'num_classes': 1,
            'num_epochs': 10,
            'tensorboard_logdir':'..{0}logs{0}resnet_1D_with_dropout'.format(os.sep)
        }
        return params

    def train(self, trainfile, validationfile):
        # refactor train function and args
        inputshape = (self.params['batchsize'], self.params['num_features'], 1)
        model = self.resnet_with_dropout(inputshape, self.params['num_classes'])

        model.compile(optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=['accuracy', 'mse'])

        tensorboard = TensorBoar(logdir=self.params['tensorboard_logdir'])

        model.fit_generator(datagenerator(trainfile, self.params['batchsize']),
                            steps_per_epoch=self.params['num_steps'],
                            epochs=self.params['num_epochs'], 
                            verbose=1,
                            validation_data=datagenerator(validationfile),
                            validation_steps=100,
                            callbacks = [tensorboard])
        return model


class LightGBM:
    def __init__(self):
        self.model_params = self.__initparams()
        self.optima_boost_round = 300 # to use cross-validation to estimate optimum value, set this to None


    def __initparams(self):
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
    
    def get_optima_boost_round(self, train_data):
        crossvalidation = lgbm.cv(params=self.model_params,
                                train_set=train_data,
                                nfold=10,
                                num_boost_round=1000,
                                early_stopping_rounds=100,
                                verbose_eval=10,
                                categorical_feature=self.categorical_features)

    def train(self, trainfile):
        '''
        load training file and train lightgbm model
        '''
        features, labels = load_dataset(trainfile)
        self.categorical_features = features.columns.tolist()
        train_data = lgbm.Dataset(data=features, label=labels,
                                    feature_name=features.columns.tolist(),
                                    categorical_feature=self.categorical_features)

        if self.optima_boost_round is None: self.optima_boost_round = self.get_optima_boost_round(train_data)

        model = lgbm.train(params=self.model_params, 
                            train_set=train_data, 
                            num_boost_round=self.optima_boost_round, 
                            categorical_feature=self.categorical_features)
        return model