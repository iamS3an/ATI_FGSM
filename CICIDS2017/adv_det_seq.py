import tensorflow as tf
import keras
from keras import backend, losses
from keras.models import load_model, Model, load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
import time
import os
import random


seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def get_prediction(model, inputs, label):
    predictions = model.predict(inputs)
    pre_label = pd.DataFrame(predictions)
    # print(inputs, pre_label)
    fpr, tpr, thresholds = roc_curve(label, pre_label, pos_label=1)
    auc_scores = auc(fpr, tpr)
    rounded_pre_label = pd.DataFrame([np.round(x) for x in predictions])
    tn, fp, fn, tp = confusion_matrix(label, rounded_pre_label).ravel()
    print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
    print(f"ACC:{(tp+tn)/(tp+tn+fp+fn)*100}%")
    print(f"AUC:{auc_scores*100}%")
    print(f"TPR:{tp/(tp+fn)*100}%")
    print(f"FPR:{fp/(tn+fp)*100}%")
    
    return predictions

status = 'origin'

origin_train = pd.read_csv(f'./{status}/train.csv', header=0, sep=',', comment='#', low_memory=False)
origin_test = pd.read_csv(f'./{status}/test.csv', header=0, comment='#', low_memory=False)

X_train = origin_train.drop(['label'], axis=1)
X_test = origin_test.drop(['label'], axis=1)

for eps in [0.05, 0.1, 0.01, 0.02]:
    train_adv_data = pd.read_csv(f'./{status}/{eps}/train_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    test_adv_data = pd.read_csv(f'./{status}/{eps}/test_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
    
    # reset column to concat
    X_train.columns = range(X_train.columns.size)
    X_test.columns = range(X_test.columns.size)
    train_adv_data.columns = range(train_adv_data.columns.size)
    test_adv_data.columns = range(test_adv_data.columns.size)

    second_train_data = pd.concat([X_train, train_adv_data])
    second_train_label = pd.concat([pd.Series(0, index=np.arange(len(X_train))), pd.Series(1, index=np.arange(len(train_adv_data)))])
    (second_train_data, second_train_label) = shuffle(second_train_data, second_train_label, random_state=0)
    second_test_data = pd.concat([X_test, test_adv_data])
    second_test_label = pd.concat([pd.Series(0, index=np.arange(len(X_test))), pd.Series(1, index=np.arange(len(test_adv_data)))])

    tf.keras.backend.clear_session()
    inputs = keras.Input(shape=(78,))
    # o = keras.layers.Dense(13, activation="LeakyReLU", name="dense_2")(inputs)
    #o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(inputs)
    
    adv_model = Model(inputs,outputs)
    adv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    adv_model.fit(second_train_data, second_train_label, batch_size=256, epochs=1)

    print("Evaluating adv data detection...")
    predictions = get_prediction(adv_model, second_test_data, second_test_label)

