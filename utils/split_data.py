import pandas as pd

DEV_FRACTION = 0.05
SEED = 4
data_folder = '../data/original/'

data = pd.read_csv(data_folder + 'Dataset_Salesforce_Predictive_Modelling_TRAIN.txt')

nr_dev_items = int(len(data)*DEV_FRACTION)

shuffled_data = data.sample(frac=1,replace=False,random_state=SEED)

dev = shuffled_data[0:nr_dev_items]
train = shuffled_data[nr_dev_items:len(data)]

dev.to_csv('dev.csv',index=False)
train.to_csv('train.csv',index=False)
