import split_TEST
import split_TRAIN
import split_VALIDATION
import pandas as pd
from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt

labelsize = 12
fontsize = 12

path_parameters = 'res_hyperopt.pkl'
df_in = pd.read_pickle('375Hz_2classes.pkl')


def get_best_parameters(df_par, par1, par2):
    if ((par1 == '') == True and (par2 == '') == False) or (par1 == par2):
        load = par2
    elif (par1 == '') == False and (par2 == '') == True:
        load = par1
    else:
        load = par1 + '_' + par2

    row_number = df_par.index[df_par['load'] == load]
    parameters = df_par.loc[row_number, 'hyperparameters'].to_list()[0]

    return parameters


loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']

df_best_parameters = pd.read_pickle(path_parameters)

av_acc_test = 0
av_prec_test = 0
av_f1_test = 0
av_rec_test = 0

for i in range(0, len(loads), 1):

    par1_test = loads[i]
    par2_test = loads[i]

    dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
    dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.2)
    dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

    feature_names = df_in.columns.to_list()[1:len(df_in.columns) - 1]  # first element is the name, the last is 'D'

    X_val = dataset_validation[feature_names]
    y_val = dataset_validation['D_class']

    X_train = dataset_train[feature_names]
    y_train = dataset_train['D_class']

    X_test = dataset_test[feature_names]
    y_test = dataset_test['D_class']

    best_parameters = get_best_parameters(df_best_parameters, par1_test, par2_test)

    clf = tree.DecisionTreeClassifier(**best_parameters, random_state=0)

    clf.fit(X_train, y_train)

    p_train = clf.predict(X_train)
    p_test = clf.predict(X_test)

    acc_test = metrics.accuracy_score(y_test, p_test)
    rec = metrics.recall_score(y_test, p_test)
    f1 = metrics.f1_score(y_test, p_test)
    prec = metrics.precision_score(y_test, p_test)

    print('\n************************************************************')
    print(f'{loads[i]}) Test accuracy: {round(acc_test * 100)} %')
    print(f'{loads[i]}) Test precision: {round(prec * 100)} %')
    print(f'{loads[i]}) Test recall: {round(rec * 100)} %')
    print(f'{loads[i]}) Test F1: {round(f1 * 100)} %')

    av_acc_test = av_acc_test + acc_test
    av_f1_test = av_f1_test + f1
    av_prec_test = av_prec_test + prec
    av_rec_test = av_rec_test + rec

print('\n************************************************************')
print(f'Average test accuracy: {round(av_acc_test/len(loads) * 100, 1)} %')
print(f'Average test precision: {round(av_prec_test/len(loads) * 100, 1)} %')
print(f'Average test recall: {round(av_rec_test/len(loads) * 100, 1)} %')
print(f'Average test F1: {round(av_f1_test/len(loads) * 100, 1)} %')
