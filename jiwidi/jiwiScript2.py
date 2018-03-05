import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import shutil
import sys

parser = argparse.ArgumentParser()




#Read data
TRAIN = pd.read_csv('../data/train.csv')
TRAIN.fillna('', inplace=True)
DEV = pd.read_csv('../data/dev.csv')
DEV.fillna('', inplace=True)
PREDICT = pd.read_csv('../data/original/Dataset_Salesforce_Predictive_Modelling_TEST.txt')
#Read target data
TARGET_TRAIN = TRAIN["Poder_Adquisitivo"]#.as_matrix()
TARGET_DEV = DEV["Poder_Adquisitivo"]#.as_matrix()


#Drop target data from training data and id
TRAIN = TRAIN.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)
DEV = DEV.drop("ID_Customer", axis=1).drop("Poder_Adquisitivo", axis=1)



_CSV_COLUMNS =list(TRAIN.columns)


_CSV_COLUMN_DEFAULTS = []

for type in TRAIN.dtypes:
    if type=='object':
        _CSV_COLUMN_DEFAULTS.append([''])
    else:
        _CSV_COLUMN_DEFAULTS.append([1])

parser.add_argument(
    '--model_dir', type=str, default='/tmp/census_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide',
    help="Valid model types: {'wide deep wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='../data/train3.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='../data/dev3.csv',
    help='Path to the test data.')

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  Imp_Cons_01= tf.feature_column.numeric_column('Imp_Cons_01')
  Imp_Cons_02= tf.feature_column.numeric_column('Imp_Cons_02')
  Imp_Cons_03= tf.feature_column.numeric_column('Imp_Cons_03')
  Imp_Cons_04= tf.feature_column.numeric_column('Imp_Cons_04')
  Imp_Cons_05= tf.feature_column.numeric_column('Imp_Cons_05')
  Imp_Cons_06= tf.feature_column.numeric_column('Imp_Cons_06')
  Imp_Cons_07= tf.feature_column.numeric_column('Imp_Cons_07')
  Imp_Cons_08= tf.feature_column.numeric_column('Imp_Cons_08')
  Imp_Cons_09= tf.feature_column.numeric_column('Imp_Cons_09')
  Imp_Cons_10= tf.feature_column.numeric_column('Imp_Cons_10')
  Imp_Cons_11= tf.feature_column.numeric_column('Imp_Cons_11')
  Imp_Cons_12= tf.feature_column.numeric_column('Imp_Cons_12')
  Imp_Cons_13= tf.feature_column.numeric_column('Imp_Cons_13')
  Imp_Cons_14= tf.feature_column.numeric_column('Imp_Cons_14')
  Imp_Cons_15= tf.feature_column.numeric_column('Imp_Cons_15')
  Imp_Cons_16= tf.feature_column.numeric_column('Imp_Cons_16')
  Imp_Cons_17= tf.feature_column.numeric_column('Imp_Cons_17')
  Imp_Sal_01= tf.feature_column.numeric_column('Imp_Sal_01')
  Imp_Sal_02= tf.feature_column.numeric_column('Imp_Sal_02')
  Imp_Sal_03= tf.feature_column.numeric_column('Imp_Sal_03')
  Imp_Sal_04= tf.feature_column.numeric_column('Imp_Sal_04')
  Imp_Sal_05= tf.feature_column.numeric_column('Imp_Sal_05')
  Imp_Sal_06= tf.feature_column.numeric_column('Imp_Sal_06')
  Imp_Sal_07= tf.feature_column.numeric_column('Imp_Sal_07')
  Imp_Sal_08= tf.feature_column.numeric_column('Imp_Sal_08')
  Imp_Sal_09= tf.feature_column.numeric_column('Imp_Sal_09')
  Imp_Sal_10= tf.feature_column.numeric_column('Imp_Sal_10')
  Imp_Sal_11= tf.feature_column.numeric_column('Imp_Sal_11')
  Imp_Sal_12= tf.feature_column.numeric_column('Imp_Sal_12')
  Imp_Sal_13= tf.feature_column.numeric_column('Imp_Sal_13')
  Imp_Sal_14= tf.feature_column.numeric_column('Imp_Sal_14')
  Imp_Sal_15= tf.feature_column.numeric_column('Imp_Sal_15')
  Imp_Sal_16= tf.feature_column.numeric_column('Imp_Sal_16')
  Imp_Sal_17= tf.feature_column.numeric_column('Imp_Sal_17')
  Imp_Sal_18= tf.feature_column.numeric_column('Imp_Sal_18')
  Imp_Sal_19= tf.feature_column.numeric_column('Imp_Sal_19')
  Imp_Sal_20= tf.feature_column.numeric_column('Imp_Sal_20')
  Imp_Sal_21= tf.feature_column.numeric_column('Imp_Sal_21')
  Ind_Prod_01= tf.feature_column.numeric_column('Ind_Prod_01')
  Ind_Prod_02= tf.feature_column.numeric_column('Ind_Prod_02')
  Ind_Prod_03= tf.feature_column.numeric_column('Ind_Prod_03')
  Ind_Prod_04= tf.feature_column.numeric_column('Ind_Prod_04')
  Ind_Prod_05= tf.feature_column.numeric_column('Ind_Prod_05')
  Ind_Prod_06= tf.feature_column.numeric_column('Ind_Prod_06')
  Ind_Prod_07= tf.feature_column.numeric_column('Ind_Prod_07')
  Ind_Prod_08= tf.feature_column.numeric_column('Ind_Prod_08')
  Ind_Prod_09= tf.feature_column.numeric_column('Ind_Prod_09')
  Ind_Prod_10= tf.feature_column.numeric_column('Ind_Prod_10')
  Ind_Prod_11= tf.feature_column.numeric_column('Ind_Prod_11')
  Ind_Prod_12= tf.feature_column.numeric_column('Ind_Prod_12')
  Ind_Prod_13= tf.feature_column.numeric_column('Ind_Prod_13')
  Ind_Prod_14= tf.feature_column.numeric_column('Ind_Prod_14')
  Ind_Prod_15= tf.feature_column.numeric_column('Ind_Prod_15')
  Ind_Prod_16= tf.feature_column.numeric_column('Ind_Prod_16')
  Ind_Prod_17= tf.feature_column.numeric_column('Ind_Prod_17')
  Ind_Prod_18= tf.feature_column.numeric_column('Ind_Prod_18')
  Ind_Prod_19= tf.feature_column.numeric_column('Ind_Prod_19')
  Ind_Prod_20= tf.feature_column.numeric_column('Ind_Prod_20')
  Ind_Prod_21= tf.feature_column.numeric_column('Ind_Prod_21')
  Ind_Prod_22= tf.feature_column.numeric_column('Ind_Prod_22')
  Ind_Prod_23= tf.feature_column.numeric_column('Ind_Prod_23')
  Ind_Prod_24= tf.feature_column.numeric_column('Ind_Prod_24')
  Num_Oper_01= tf.feature_column.numeric_column('Num_Oper_01')
  Num_Oper_02= tf.feature_column.numeric_column('Num_Oper_02')
  Num_Oper_03= tf.feature_column.numeric_column('Num_Oper_03')
  Num_Oper_04= tf.feature_column.numeric_column('Num_Oper_04')
  Num_Oper_05= tf.feature_column.numeric_column('Num_Oper_05')
  Num_Oper_06= tf.feature_column.numeric_column('Num_Oper_06')
  Num_Oper_07= tf.feature_column.numeric_column('Num_Oper_07')
  Num_Oper_08= tf.feature_column.numeric_column('Num_Oper_08')
  Num_Oper_09= tf.feature_column.numeric_column('Num_Oper_09')
  Num_Oper_10= tf.feature_column.numeric_column('Num_Oper_10')
  Num_Oper_11= tf.feature_column.numeric_column('Num_Oper_11')
  Num_Oper_12= tf.feature_column.numeric_column('Num_Oper_12')
  Num_Oper_13= tf.feature_column.numeric_column('Num_Oper_13')
  Num_Oper_14= tf.feature_column.numeric_column('Num_Oper_14')
  Num_Oper_15= tf.feature_column.numeric_column('Num_Oper_15')
  Num_Oper_16= tf.feature_column.numeric_column('Num_Oper_16')
  Num_Oper_17= tf.feature_column.numeric_column('Num_Oper_17')
  Num_Oper_18= tf.feature_column.numeric_column('Num_Oper_18')
  Num_Oper_19= tf.feature_column.numeric_column('Num_Oper_19')
  Num_Oper_20= tf.feature_column.numeric_column('Num_Oper_20')
  Socio_Demo_01= tf.feature_column.categorical_column_with_hash_bucket(
      'Socio_Demo_01', hash_bucket_size=1000)
  Socio_Demo_02= tf.feature_column.numeric_column('Socio_Demo_02')
  Socio_Demo_03= tf.feature_column.numeric_column('Socio_Demo_03')
  Socio_Demo_04= tf.feature_column.numeric_column('Socio_Demo_04')
  Socio_Demo_05= tf.feature_column.numeric_column('Socio_Demo_05')

  wide_columns = [ Imp_Cons_01, Imp_Cons_02, Imp_Cons_03,
       Imp_Cons_04, Imp_Cons_05, Imp_Cons_06, Imp_Cons_07,
       Imp_Cons_08, Imp_Cons_09, Imp_Cons_10, Imp_Cons_11,
       Imp_Cons_12, Imp_Cons_13, Imp_Cons_14, Imp_Cons_15,
       Imp_Cons_16, Imp_Cons_17, Imp_Sal_01, Imp_Sal_02, Imp_Sal_03,
       Imp_Sal_04, Imp_Sal_05, Imp_Sal_06, Imp_Sal_07, Imp_Sal_08,
       Imp_Sal_09, Imp_Sal_10, Imp_Sal_11, Imp_Sal_12, Imp_Sal_13,
       Imp_Sal_14, Imp_Sal_15, Imp_Sal_16, Imp_Sal_17, Imp_Sal_18,
       Imp_Sal_19, Imp_Sal_20, Imp_Sal_21, Ind_Prod_01, Ind_Prod_02,
       Ind_Prod_03, Ind_Prod_04, Ind_Prod_05, Ind_Prod_06,
       Ind_Prod_07, Ind_Prod_08, Ind_Prod_09, Ind_Prod_10,
       Ind_Prod_11, Ind_Prod_12, Ind_Prod_13, Ind_Prod_14,
       Ind_Prod_15, Ind_Prod_16, Ind_Prod_17, Ind_Prod_18,
       Ind_Prod_19, Ind_Prod_20, Ind_Prod_21, Ind_Prod_22,
       Ind_Prod_23, Ind_Prod_24, Num_Oper_01, Num_Oper_02,
       Num_Oper_03, Num_Oper_04, Num_Oper_05, Num_Oper_06,
       Num_Oper_07, Num_Oper_08, Num_Oper_09, Num_Oper_10,
       Num_Oper_11, Num_Oper_12, Num_Oper_13, Num_Oper_14,
       Num_Oper_15, Num_Oper_16, Num_Oper_17, Num_Oper_18,
       Num_Oper_19, Num_Oper_20, Socio_Demo_01, Socio_Demo_02,
       Socio_Demo_03, Socio_Demo_04, Socio_Demo_05
  ]
  return wide_columns, wide_columns

def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    print('Wide estimator')
    return tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    print('Deep estimator')
    return tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    print('Wide & deep estimator')
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""

  def parse_csv(value):
      print('Parsing', data_file)
      columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
      features = dict(zip(_CSV_COLUMNS, columns))
      labels = features.pop('Poder_Adquisitivo')
      return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset('../data/train3.csv')

  #if shuffle:
  #  dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

def input_fn_train(): # returns x, y
    dict={}

    for column in list(TRAIN.columns):
        dict[column]=tf.convert_to_tensor((TRAIN[column]))

    return dict,tf.convert_to_tensor(np.array(TARGET_TRAIN))


def input_fn_eval(): # returns x, y
    dict = {}

    for column in list(DEV.columns):
        dict[column] = tf.convert_to_tensor(np.array(DEV[column]))

    return dict, tf.convert_to_tensor(np.array(TARGET_DEV))

def input_fn_evaluate():
    dict = {}

    for column in list(PREDICT.columns):
        dict[column] = tf.convert_to_tensor(np.array(PREDICT[column]))

    return dict
def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn_train())

    results = model.evaluate(input_fn=lambda: input_fn_eval())

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)


    for key in sorted(results):
      print('%s: %s' % (key, results[key]))

    print('loss he' + results["loss"])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

