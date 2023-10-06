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
Y_train = origin_train.loc[:, ['label']]
X_test = origin_test.drop(['label'], axis=1)
Y_test = origin_test.loc[:, ['label']]

eps = 0.1

tf.keras.backend.clear_session()
inputs = keras.Input(shape=(78,))
o = keras.layers.Dense(64, activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(o)

origin_model = Model(inputs, outputs)
origin_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
origin_model.fit(X_train, Y_train, batch_size=256, epochs=1)

print("Evaluating normal data...")
get_prediction(origin_model, X_test, Y_test)

train_adv_data = pd.read_csv(f'./{status}/{eps}/train_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)
test_adv_data = pd.read_csv(f'./{status}/{eps}/test_adv_data.csv', header=None, sep=',', comment='#', low_memory=False)

print("Evaluating adv data...")
get_prediction(origin_model, test_adv_data, Y_test)

# reset column to concat
X_train.columns = range(X_train.columns.size)
train_adv_data.columns = range(train_adv_data.columns.size)
X_test.columns = range(X_test.columns.size)
test_adv_data.columns = range(test_adv_data.columns.size)

combined_train_data = pd.concat([X_train, train_adv_data])
combined_train_label = pd.concat([pd.Series(0, index=np.arange(len(X_train))), pd.Series(1, index=np.arange(len(train_adv_data)))])
(combined_train_data, combined_train_label) = shuffle(combined_train_data, combined_train_label, random_state=0)
combined_test_data = pd.concat([X_test, test_adv_data])
combined_test_label = pd.concat([pd.Series(0, index=np.arange(len(X_test))), pd.Series(1, index=np.arange(len(test_adv_data)))])
benign_malicious_label = pd.concat([Y_test, Y_test])

print("Evaluating normal adv combined data...")
get_prediction(origin_model, combined_test_data, benign_malicious_label)

tf.keras.backend.clear_session()
inputs = keras.Input(shape=(78,))
# o = keras.layers.Dense(64, activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(inputs)

adv_model = Model(inputs, outputs)
adv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
adv_model.fit(combined_train_data, combined_train_label, batch_size=256, epochs=1)

print("Generating det model prediction...")
predictions = get_prediction(adv_model, combined_test_data, combined_test_label)
pre_label = pd.DataFrame([np.round(x) for x in predictions])

# select the adv and drop from prediction to get normal only prediction
pre_adv_idx = pre_label.loc[(pre_label[0] == 1) & (pre_label.index >= len(X_test))].index
pre_nor_data = combined_test_data.reset_index(drop=True).drop(index=pre_adv_idx)
pre_nor_label = benign_malicious_label.reset_index(drop=True).drop(index=pre_adv_idx)

print("Evaluating recovery data...")
get_prediction(origin_model, pre_nor_data, pre_nor_label)

# ATI recovery
ATI_train = pd.read_csv(f'./adv_det_doneATI/{eps}/train.csv', header=0, sep=',', comment='#', low_memory=False)
ATI_test = pd.read_csv(f'./adv_det_doneATI/{eps}/pvalue_test.csv', header=0, comment='#', low_memory=False)

ATI_X_train = ATI_train.drop(['label'], axis=1)
ATI_Y_train = ATI_train.loc[:, ['label']]
ATI_X_test = ATI_test.drop(['label'], axis=1)
ATI_Y_test = ATI_test.loc[:, ['label']]

(ATI_X_train, ATI_Y_train) = shuffle(ATI_X_train, ATI_Y_train, random_state=0)

keras.backend.clear_session()
inputs = keras.Input(shape=(78,))
# o = keras.layers.Dense(64, activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(inputs)

ATI_model = Model(inputs, outputs)
ATI_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
ATI_model.fit(ATI_X_train, ATI_Y_train, batch_size=256, epochs=1)

print("Generating ATI det model prediction...")
predictions = get_prediction(ATI_model, ATI_X_test, ATI_Y_test)
pre_label = pd.DataFrame([np.round(x) for x in predictions])

# select the adv and drop from prediction to get normal only prediction
pre_adv_idx = pre_label.loc[(pre_label[0] == 1) & (pre_label.index >= len(X_test))].index
pre_nor_data = combined_test_data.reset_index(drop=True).drop(index=pre_adv_idx)
pre_nor_label = benign_malicious_label.reset_index(drop=True).drop(index=pre_adv_idx)

print("Evaluating ATI recovery data...")
get_prediction(origin_model, pre_nor_data, pre_nor_label)

