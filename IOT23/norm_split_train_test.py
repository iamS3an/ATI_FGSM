import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import random


seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

origin_data = pd.read_csv('combined_iot23_normalize.csv', header=None, sep=',', comment='#', low_memory=False)
origin_label = pd.read_csv('combined_iot23_label.csv', names=['label'], comment='#', low_memory=False)
# origin_data = pd.read_csv('small_pre_iot23_normalize.csv', header=None, sep=',', comment='#', low_memory=False)
# origin_label = pd.read_csv('small_pre_iot23_label.csv', names=['label'], comment='#', low_memory=False)

print(origin_data)
print(origin_label)

(X_train, X_test, Y_train, Y_test) = train_test_split(origin_data, origin_label, test_size=0.2, random_state=0)
# (X_train, Y_train) = shuffle(origin_data, origin_label, random_state=0)

train = pd.concat([X_train, Y_train], axis=1)
test = pd.concat([X_test, Y_test], axis=1)

print(train)
print(test)

train.to_csv('./origin/train.csv', index=False)
test.to_csv('./origin/test.csv', index=False)
