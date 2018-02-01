import pandas as pd
import numpy as np
import tensorflow as tf
import csv


LEARNING_RATE = 0.001
DROPOUT = 0.1
EPOCH = 5

TRAIN = pd.read_csv('../data/train.txt')
TEST = pd.read_csv('../data/test.txt')
TARGET = []
TARGET = TRAIN["Poder_Adquisitivo"].as_matrix()
TRAIN = TRAIN.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)
TEST = TEST.drop("ID_Customer", axis=1)

categorical = ["Socio_Demo_01", "Socio_Demo_02", "Socio_Demo_03", "Socio_Demo_04", "Socio_Demo_05"]
for l in categorical:
    TRAIN = TRAIN.drop(l, axis=1)
    TEST = TEST.drop(l, axis=1)

def main():
    # Deep Neural Network Regressor with the training set which contain the data split by train test split
    prepro()
    regressor = model()
    regressor.fit(input_fn=lambda: input_fn_train(), steps=100000)


    # Evaluation on the test set created by train_test_split
    ev = regressor.evaluate(input_fn=lambda: input_fn_train(), steps=1)
    loss_score1 = ev["loss"]
    print("Final Loss on the testing set: {0:f}".format(loss_score1))

def model():
    feature_cols = [tf.contrib.layers.real_valued_column("", dimension=2)]

    # Model
    tf.logging.set_verbosity(tf.logging.ERROR)
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              model_dir='model',
                                              activation_fn=tf.nn.relu,
                                              hidden_units=[200, 100, 50, 25, 12],
                                              dropout=0.1,
                                              optimizer=tf.train.ProximalAdagradOptimizer(
                                                learning_rate=0.1,
                                                l1_regularization_strength=0.001
                                                )
    )


    return regressor

def prepro():
    global TRAIN
    global TEST




# Input builders
def input_fn_train(): # returns x, y
    return tf.convert_to_tensor(np.array(TRAIN[:60])),tf.convert_to_tensor(np.array(TARGET[:60]))


def input_fn_eval(): # returns x, y
    return tf.convert_to_tensor(np.array(TRAIN[60:])), tf.convert_to_tensor(np.array(TARGET[60:]))

#metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
def input_fn_predict(): # returns x, None
    return tf.convert_to_tensor(np.array(TEST))

if __name__ == "__main__":
  main()










