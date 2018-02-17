import pandas as pd
import numpy as np
import tensorflow as tf
import csv


LEARNING_RATE = 0.001
DROPOUT = 0.1
EPOCH = 10000

TRAIN = pd.read_csv('../data/train.csv')
DEV = pd.read_csv('../data/dev.csv')
TARGET = []
TARGET_TRAIN = TRAIN["Poder_Adquisitivo"].as_matrix()
TARGET_DEV = DEV["Poder_Adquisitivo"].as_matrix()
TRAIN = TRAIN.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)
DEV = DEV.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)

categorical = ["Socio_Demo_01", "Socio_Demo_02", "Socio_Demo_03", "Socio_Demo_04", "Socio_Demo_05"]
for l in categorical:
    TRAIN = TRAIN.drop(l, axis=1)
    DEV = DEV.drop(l, axis=1)

def main():
    # Deep Neural Network Regressor with the training set which contain the data split by train test split
    prepro()


    regressor = model(EPOCH,DROPOUT,LEARNING_RATE)
    regressor.fit(input_fn=lambda: input_fn_train(), steps=EPOCH)
    # Evaluation on the test set created by train_test_split
    ev = regressor.evaluate(input_fn=lambda: input_fn_eval(), steps=1)
    loss_score1 = ev["loss"]
    print('E'+str(EPOCH)+'-D'+str(DROPOUT)+'-L'+str(LEARNING_RATE)+" Final Loss on the testing set: {0:f}".format(loss_score1))


def model(EPOCH,DROPOUT,LEARNING_RATE):
    feature_cols = [tf.contrib.layers.real_valued_column("", dimension=2)]

    # Model
    tf.logging.set_verbosity(tf.logging.ERROR)
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              model_dir='model\model-E'+str(EPOCH)+'-D'+str(DROPOUT)+'-L'+str(LEARNING_RATE),
                                              activation_fn=tf.nn.relu,
                                              hidden_units=[200, 100, 50, 25, 12],
                                              dropout=DROPOUT,
                                              optimizer=tf.train.ProximalAdagradOptimizer(
                                                learning_rate=LEARNING_RATE,
                                                l1_regularization_strength=0.001
                                                )
    )


    return regressor

def prepro():
    global TRAIN
    global TEST




# Input builders
def input_fn_train(): # returns x, y
    return tf.convert_to_tensor(np.array(TRAIN)),tf.convert_to_tensor(np.array(TARGET_TRAIN))


def input_fn_eval(): # returns x, y
    return tf.convert_to_tensor(np.array(DEV)), tf.convert_to_tensor(np.array(TARGET_DEV))

#metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
def input_fn_predict(): # returns x, None
    return tf.convert_to_tensor(np.array(TEST))

main()
