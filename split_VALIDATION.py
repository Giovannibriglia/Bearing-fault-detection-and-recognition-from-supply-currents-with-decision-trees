from sklearn.model_selection import train_test_split


def setVal(dataset, val_size=0.2):

    df_train, df_val = train_test_split(dataset, test_size=val_size, random_state=0)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    return df_val, df_train
