import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time


origin_train = pd.read_csv(f'./origin/train.csv', header=0, sep=',', comment='#', low_memory=False)
origin_test = pd.read_csv(f'./origin/test.csv', header=0, comment='#', low_memory=False)

X_train = origin_train.drop(['label'], axis=1)
X_test = origin_test.drop(['label'], axis=1)

# X_train = origin_train
# X_test = origin_test

for eps in [0.05, 0.1, 0.01, 0.02]:
    train_adv_data = pd.read_csv(f'./origin/{eps}/train_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    test_adv_data = pd.read_csv(f'./origin/{eps}/test_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    # train_adv_label = pd.read_csv(f'./origin/{eps}/train_adv_label.csv', names=['label'], comment='#', low_memory=False)
    # test_adv_label = pd.read_csv(f'./origin/{eps}/test_adv_label.csv', names=['label'], comment='#', low_memory=False)

    # train_adv_data = pd.concat([train_adv_data, train_adv_label], axis=1)
    # test_adv_data = pd.concat([test_adv_data, test_adv_label], axis=1)

    col_num = list(range(X_train.columns.size))
    
    X_train.columns = col_num
    train_adv_data.columns = col_num
    X_test.columns = col_num
    test_adv_data.columns = col_num
    
    combined_train = pd.concat([X_train, train_adv_data])
    train_label = pd.concat([pd.Series(0, index=np.arange(len(X_train))), pd.Series(1, index=np.arange(len(train_adv_data)))])
    combined_test = pd.concat([X_test, test_adv_data])
    test_label = pd.concat([pd.Series(0, index=np.arange(len(X_test))), pd.Series(1, index=np.arange(len(test_adv_data)))])
    
    train = pd.concat([combined_train, train_label], axis=1)
    test = pd.concat([combined_test, test_label], axis=1)
    
    col_num_with_label = list(range(combined_train.columns.size)) + ['label']
    train.columns = col_num_with_label
    test.columns = col_num_with_label
    
    print(train)
    print(test)

    train.to_csv(f'./adv_det_preATI/{eps}/train.csv', index=False)
    test.to_csv(f'./adv_det_preATI/{eps}/test.csv', index=False)
    