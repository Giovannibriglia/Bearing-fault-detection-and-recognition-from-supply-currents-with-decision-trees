import pandas as pd


def TRAIN(dataset, par1, par2):

    df_train = pd.DataFrame([[0] * len(dataset.columns)], columns=dataset.columns)

    col_name = dataset['name_signal']

    for i in range(0, len(dataset), 1):

        if (par1 == '') == True and (par2 == '') == False:
            set = not(col_name[i].find(par2) >= 0)

        elif (par1 == '') == False and (par2 == '') == True:
            set = not(col_name[i].find(par1) >= 0)

        else:
            set = not(col_name[i].find(par2) >= 0 and col_name[i].find(par1) >= 0)

        if set:
            row = dataset.loc[[i]]

            df_train = df_train._append(row, ignore_index=True)

    df_train.drop(0, inplace=True)
    return df_train






