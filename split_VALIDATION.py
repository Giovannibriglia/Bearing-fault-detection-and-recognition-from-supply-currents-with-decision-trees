import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

seed_value = 42
random.seed(seed_value)


def setVal(dataset, val_size=0.2):

    df_train, df_val = train_test_split(dataset, test_size=val_size, random_state=seed_value)

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    return df_val, df_train