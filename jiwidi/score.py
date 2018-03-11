import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
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

def input_fn_eval(): # returns x, y
    dict = {}

    for column in list(DEV.columns):
        dict[column] = tf.convert_to_tensor(np.array(DEV[column]))

    return dict, tf.convert_to_tensor(np.array(TARGET_DEV))

def input_fn_predict():
    dict = {}

    for column in list(DEV.columns):
        dict[column] = tf.convert_to_tensor(np.array(DEV[column]))

    return dict

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
       Imp_Sal_19, Imp_Sal_20, Imp_Sal_21
  ]
  deep_columns = [
      Ind_Prod_01, Ind_Prod_02,
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
      Num_Oper_19, Num_Oper_20, Socio_Demo_02,
      Socio_Demo_03, Socio_Demo_04, Socio_Demo_05,
      tf.feature_column.indicator_column(Socio_Demo_01),
  ]
  return wide_columns, deep_columns

def train_and_evaluate(model, splits, skcompat=False, scaler=None):
    rmse = []
    mae = []
    mad = []
    # Para cada iteración de validación cruzada
    #for s in range(len(splits)):



    yhat = (model.predict(input_fn=lambda: input_fn_predict()))

    results=[]
    i=1
    for y in yhat:
        i = i + 1
        results.append(y['predictions'][0])
        print(y['predictions'][0])
        if(i>18191):
            print('hi'
                  )
            break
    print(results)



    # Calculamos métricas
    rmse.append(math.sqrt(mean_squared_error(y_true=TARGET_DEV, y_pred=results)))
    mae.append(mean_absolute_error(y_true=TARGET_DEV, y_pred=results))
    mad.append(median_absolute_error(y_true=TARGET_DEV, y_pred=results))

    return (rmse, mae, mad)

def model_jiwi():
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto())#device_count={'GPU': 0}))

    # Model
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir='C:\\Users\\jiwidi\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\wide_deep',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        #fix_global_step_increment_bug=True,
        dnn_dropout=0.1,
    )
model=model_jiwi()

scores_rmse,scores_mae,scores_mad = train_and_evaluate(model,None)


print("RMSE: %f" % np.mean(scores_rmse))
print("MAE: %f" % np.mean(scores_mae))
print("MAD: %f" % np.mean(scores_mad))