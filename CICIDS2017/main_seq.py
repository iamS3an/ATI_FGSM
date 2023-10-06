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

def gen_adv(model, inputs, label):
    model.evaluate(inputs, label, batch_size=256)
    adv_data = pd.read_csv('adv_data_temp.csv', header=None, sep=',', comment='#', low_memory=False)
    pd.DataFrame().to_csv('adv_data_temp.csv', index=False, header=False, mode='w')
    for col in adv_data.columns:
        adv_data.loc[adv_data[col] > 1, col] = 1
        adv_data.loc[adv_data[col] < 0, col] = 0
    
    return adv_data

class CustomModel(keras.Model):
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self(x, training=False)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        signed_grad = tf.transpose(backend.sign(gradients[0]))
        adv = x + tf.cast(signed_grad, tf.float64)[0] * eps
        pd.DataFrame(adv.numpy()).to_csv('adv_data_temp.csv', index=False, header=False, mode='a')
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

tf.keras.backend.clear_session()
inputs = keras.Input(shape=(78,))
o = keras.layers.Dense(64, activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(o)

status = 'ATI'

origin_train = pd.read_csv(f'./{status}/train.csv', header=0, sep=',', comment='#', low_memory=False)
origin_test = pd.read_csv(f'./{status}/test.csv', header=0, comment='#', low_memory=False)

X_train = origin_train.drop(['label'], axis=1)
Y_train = origin_train.loc[:, ['label']]
X_test = origin_test.drop(['label'], axis=1)
Y_test = origin_test.loc[:, ['label']]

model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(X_train, Y_train, batch_size=256, epochs=1)

print("Evaluating normal data...")
get_prediction(model, X_test, Y_test)

pd.DataFrame().to_csv('adv_data_temp.csv', index=False, header=False, mode='w')

# fgsm
for eps in [0.05, 0.1]:
    print("Generating train adv data...")
    train_adv_data = gen_adv(model, X_train, Y_train)
    train_adv_data.to_csv(f'./{status}/{eps}/train_adv_data.csv', index=False, header=False)
    print("Evaluating train adv data...")
    predictions = get_prediction(model, train_adv_data, Y_train)
    print("Generating train adv label...")
    train_adv_label = pd.DataFrame([np.round(x) for x in predictions])
    train_adv_label.to_csv(f'./{status}/{eps}/train_adv_label.csv',index=False, header=False)
        
    print("Generating test adv data...")
    test_adv_data = gen_adv(model, X_test, Y_test)
    test_adv_data.to_csv(f'./{status}/{eps}/test_adv_data.csv', index=False, header=False)
    print("Evaluating test adv data...")
    predictions = get_prediction(model, test_adv_data, Y_test)
    print("Generating test adv label...")
    test_adv_label = pd.DataFrame([np.round(x) for x in predictions])
    test_adv_label.to_csv(f'./{status}/{eps}/test_adv_label.csv',index=False, header=False)

# ifgsm
for eps in [0.01, 0.02]:
    print("Generating train adv data...")
    train_adv_data = X_train
    for i in range(5):
        train_adv_data = gen_adv(model, train_adv_data, Y_train)
    train_adv_data.to_csv(f'./{status}/{eps}/train_adv_data.csv', index=False, header=False)
    print("Evaluating train adv data...")
    predictions = get_prediction(model, train_adv_data, Y_train)
    print("Generating train adv label...")
    train_adv_label = pd.DataFrame([np.round(x) for x in predictions])
    train_adv_label.to_csv(f'./{status}/{eps}/train_adv_label.csv',index=False, header=False)
    
    print("Generating test adv data...")
    test_adv_data = X_test
    for i in range(5):
        test_adv_data = gen_adv(model, test_adv_data, Y_test)    
    test_adv_data.to_csv(f'./{status}/{eps}/test_adv_data.csv', index=False, header=False)
    print("Evaluating test adv data...")
    predictions = get_prediction(model, test_adv_data, Y_test)
    print("Generating test adv label...")
    test_adv_label = pd.DataFrame([np.round(x) for x in predictions])
    test_adv_label.to_csv(f'./{status}/{eps}/test_adv_label.csv',index=False, header=False)


