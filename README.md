# AdTracker
AdTracking Fraud Detection -  detect fraudulent click traffic for ads
<link>https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection</link>

#### Usage
<code>
from model import LightGBM, ResNet
import os
datapath = '..{}data'.format(os.sep)
trainfile = '{}sample_train_data.csv'.format(datapath)
valfile = '{}sample_val_data.csv'.format(datapath)
</code>

'''example using ResNet model'''
<code>
resnet = ResNet()
resnet_model = resnet.train(trainfile, valfile)
</code>
'''use trained model for prediction - resnet_model.predict()'''
'''example using LightGBM model'''
<code>
lightgbm = LightGBM()
lightgbm_model = lightgbm.train(trainfile)
</code>
<code>
'''use trained model for prediction - lightgbm_model.predict()'''
</code>
