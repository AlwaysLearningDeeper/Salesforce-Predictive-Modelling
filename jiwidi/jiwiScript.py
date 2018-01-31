import pandas as pd
import tensorflow as tf
import csv

LEARNING_RATE = 0.001
DROPOUT = 0.1
TRAIN = pd.read_csv('../data/train.txt')
TEST = pd.read_csv('../data/test.txt')
TARGET = TRAIN["Poder_Adquisitivo"].as_matrix()
TRAIN = TRAIN.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)
TEST = TEST.drop("ID_Customer", axis=1)

categorical = ["Socio_Demo_01","Socio_Demo_02","Socio_Demo_03","Socio_Demo_04","Socio_Demo_05"]
for l in categorical:
    TRAIN = TRAIN.drop(l,axis=1)
    TEST = TEST.drop(l, axis=1)

def main(unsued_argv):
    # Set model params
    model_params = {"learning_rate": LEARNING_RATE,
                    "dropout": DROPOUT}

    #Fit the model
    nn = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params)
    nn.fit(x=TRAIN.as_matrix(), y=TEST.as_matrix(), steps=200000)

def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = tf.contrib.layers.relu(features, 82)

    drop_out1 = tf.nn.dropout(first_hidden_layer, params['dropout'])  # DROP-OUT here

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(drop_out1, 50)

    drop_out2 = tf.nn.dropout(second_hidden_layer, params['dropout'])

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(drop_out2, 1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"y": predictions}

    # Calculate loss using mean squared error
    loss = tf.contrib.losses.mean_squared_error(predictions, targets)
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

    return predictions_dict, loss, train_op

if __name__ == "__main__":
  tf.app.run()




