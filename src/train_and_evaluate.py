import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

CATEGORICAL = ["Socio_Demo_02","Ind_Prod_01","Ind_Prod_02","Ind_Prod_03","Ind_Prod_04","Ind_Prod_05","Ind_Prod_06","Ind_Prod_07","Ind_Prod_08",
"Ind_Prod_09","Ind_Prod_10","Ind_Prod_11","Ind_Prod_12","Ind_Prod_13","Ind_Prod_14","Ind_Prod_15","Ind_Prod_16","Ind_Prod_17","Ind_Prod_18","Ind_Prod_19",
"Ind_Prod_20","Ind_Prod_21", "Ind_Prod_22" ,"Ind_Prod_23" ,"Ind_Prod_24" ]

SIMPLE_CATEGORICAL = ["Socio_Demo_02"]


def preprocess_data(dataframe):
    # Drop columns that are too complex for now
    dataframe = dataframe.drop(labels=["Socio_Demo_01","ID_Customer"],axis=1)

    # Convert categorical data to one-hot
    dataframe = pd.get_dummies(dataframe,columns=CATEGORICAL)

    return dataframe

def simple_preprocess_data(dataframe):
    # Drop columns that are too complex for now
    dataframe = dataframe.drop(labels=["Socio_Demo_01","ID_Customer"],axis=1)

    # Convert categorical data to one-hot
    dataframe = pd.get_dummies(dataframe,columns=SIMPLE_CATEGORICAL)

    return dataframe

def remove_outlier(df_in, col_name,fence_multiplier=1.5):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-fence_multiplier*iqr
    fence_high = q3+fence_multiplier*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def evaluate(yhat,y):
    print("MSE:")
    mse = mean_squared_error(y_true=y, y_pred=yhat)
    print(mse)
    print("RMSE:")
    rmse = math.sqrt(mse)
    print(rmse)
    print("MAE:")
    mae = mean_absolute_error(y_true=y,y_pred=yhat)
    print(mae)



if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv")
    dev_df = pd.read_csv("../data/dev.csv")

    train_df = simple_preprocess_data(train_df)
    x_train = train_df.drop(labels=["Poder_Adquisitivo"],axis=1).as_matrix()
    y_train = train_df["Poder_Adquisitivo"].as_matrix()

    train_df_clear = remove_outlier(train_df,"Poder_Adquisitivo")

    dev_df = simple_preprocess_data(dev_df)
    x_dev = dev_df.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_dev = dev_df["Poder_Adquisitivo"].as_matrix()

    print("Linear Regression:")
    model = LinearRegression()
    model.fit(X=x_train,y=y_train)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat,y=y_dev)

    """
    print("Lasso Regression:")
    model = Lasso()
    model.fit(X=x_train, y=y_train)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat, y=y_dev)
    

    print("Scaled Linear Regression:")
    model = LinearRegression()
    scaler = MinMaxScaler()
    scaler.fit(X=x_train)
    model.fit(X=scaler.transform(x_train), y=y_train)
    yhat = model.predict(X=scaler.transform(x_dev))
    evaluate(yhat=yhat, y=y_dev)
    """

    train_df_clear = remove_outlier(train_df,"Poder_Adquisitivo")
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"],axis=1).as_matrix()
    y_train_clear = train_df_clear["Poder_Adquisitivo"].as_matrix()


    print("Outlier Filtered Linear Regression:")
    model = LinearRegression()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat, y=y_dev)



    print("Outlier Filtered Linear Regression, frng=2.5:")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo",2.5)
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = train_df_clear["Poder_Adquisitivo"].as_matrix()

    model = LinearRegression()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat, y=y_dev)



    """
    print("Outlier Filtered Linear Regression, frng=3:")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo", 3.0)
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = train_df_clear["Poder_Adquisitivo"].as_matrix()

    model = LinearRegression()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat, y=y_dev)

    print("Outlier Filtered Linear Regression, frng=5:")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo", 5.0)
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = train_df_clear["Poder_Adquisitivo"].as_matrix()
    
    """

    #Strange results with this one
    print("Log-Outlier Filtered Linear Regression")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo")
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = np.log(train_df_clear["Poder_Adquisitivo"].as_matrix())
    model = LinearRegression()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = np.exp(model.predict(X=x_dev))
    evaluate(yhat=yhat, y=y_dev)

    print("Outlier Filtered MLPRegressor")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo")
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = train_df_clear["Poder_Adquisitivo"].as_matrix()
    model = MLPRegressor()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = model.predict(X=x_dev)
    evaluate(yhat=yhat, y=y_dev)

    #This one breaks too
    print("Log-Outlier Filtered MLPRegressor")
    train_df_clear = remove_outlier(train_df, "Poder_Adquisitivo")
    x_train_clear = train_df_clear.drop(labels=["Poder_Adquisitivo"], axis=1).as_matrix()
    y_train_clear = np.log(train_df_clear["Poder_Adquisitivo"].as_matrix())
    model = MLPRegressor()
    model.fit(X=x_train_clear, y=y_train_clear)
    yhat = np.exp(model.predict(X=x_dev))
    evaluate(yhat=yhat, y=y_dev)







