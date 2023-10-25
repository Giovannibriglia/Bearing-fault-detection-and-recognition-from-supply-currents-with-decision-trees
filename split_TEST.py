import pandas as pd


def TEST(dataset, par1, par2):

    df_test = pd.DataFrame([[0] * len(dataset.columns)], columns=dataset.columns)

    col_name = dataset['name_signal']

    row_number = 0

    for i in range(0, len(dataset), 1):

        if (par1 == '') == True and (par2 == '') == False:
            set = col_name[i].find(par2) >= 0

        elif (par1 == '') == False and (par2 == '') == True:
            set = col_name[i].find(par1) >= 0

        else:
            set = col_name[i].find(par2) >= 0 and col_name[i].find(par1) >= 0

        if set:
            row = dataset.loc[[i]]

            df_test = df_test._append(row, ignore_index=True)

            row_number = row_number + 1

    df_test.drop(0, inplace=True)
    return df_test

