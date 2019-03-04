# AdTracker
AdTracking Fraud Detection -  detect fraudulent click traffic for ads
<link>https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection</link>

#### Usage
```python
from model import LightGBM, ResNet
import os
datapath = '..{}data'.format(os.sep)
trainfile = '{}sample_train_data.csv'.format(datapath)
valfile = '{}sample_val_data.csv'.format(datapath)

# example using ResNet model
resnet = ResNet()
resnet_model = resnet.train(trainfile, valfile)
# use trained model for prediction - resnet_model.predict()

# example using LightGBM model
lightgbm = LightGBM()
lightgbm_model = lightgbm.train(trainfile)
# use trained model for prediction - lightgbm_model.predict()
```
