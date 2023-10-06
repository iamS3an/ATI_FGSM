import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time


origin_train = pd.read_csv(f'./origin/train.csv', header=0, sep=',', comment='#', low_memory=False)
origin_test = pd.read_csv(f'./origin/test.csv', header=0, comment='#', low_memory=False)

Y_train = origin_train.loc[:, ['label']]
Y_test = origin_test.loc[:, ['label']]

for eps in [0.05, 0.1, 0.01, 0.02]:
    train_adv_data = pd.read_csv(f'./origin/{eps}/train_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    test_adv_data = pd.read_csv(f'./origin/{eps}/test_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    # train_adv_label = pd.read_csv(f'./origin/{eps}/train_adv_label.csv', names=['label'], comment='#', low_memory=False)
    # test_adv_label = pd.read_csv(f'./origin/{eps}/test_adv_label.csv', names=['label'], comment='#', low_memory=False)
    
    labeled_train_adv = pd.concat([train_adv_data, Y_train], axis=1)
    labeled_test_adv = pd.concat([test_adv_data, Y_test], axis=1)

    col_name = list(range(origin_train.columns.size - 1)) + ['label']
    
    origin_train.columns = col_name
    origin_test.columns = col_name
    labeled_train_adv.columns = col_name
    labeled_test_adv.columns = col_name
    train = pd.concat([origin_train, labeled_train_adv])
    test = pd.concat([origin_test, labeled_test_adv])
    
    print(train)
    print(test)

    train.to_csv(f'./atk_eval_preATI/{eps}/train.csv', index=False)
    test.to_csv(f'./atk_eval_preATI/{eps}/test.csv', index=False)
    