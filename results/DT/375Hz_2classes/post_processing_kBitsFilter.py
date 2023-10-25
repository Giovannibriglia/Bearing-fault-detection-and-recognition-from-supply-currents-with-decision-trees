import split_TEST
import split_TRAIN
import split_VALIDATION
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt

label_size_plot = 12
fontsize = 12
start_k, max_k = 1, 20

path_parameters = 'res_hyperopt.pkl'
df_in = pd.read_pickle('375Hz_2classes.pkl')


def get_best_parameters(df_par, par1, par2):
    if ((par1 == '') == True and (par2 == '') == False) or (par1 == par2):
        load = par2
    elif (par1 == '') == False and (par2 == '') == True:
        load = par1
    else:
        load = par1 + '_' + par2

    row_number = df_par.index[df_par['name'] == load]
    parameters = df_par.loc[row_number, 'hyperparameters'].to_list()[0]

    return parameters


def evaluate_undamaged(predictions, k):

    for i in range(1, int(len(predictions) / 3), 1):
        if i - k < 0 or i + k > len(predictions) / 3:
            add = 0
        else:
            add = k

        if predictions[i + k] == 0 and predictions[i - add] == 0:
            predictions[i] = 0

    return predictions


def evaluate_damaged(predictions, k):

    for i in range(int(len(predictions) / 3), len(predictions), 1):
        if i - k < len(predictions) / 3 or i + k > len(predictions) - 1:
            add = 0
        else:
            add = k

        if predictions[i - add] == 1 and predictions[i + add] == 1:
            predictions[i] = 1

    return predictions


loads = ['R1', 'R2', 'R3',
         'T1', 'T2', 'T3',
         'R1_T1', 'R1_T2', 'R1_T3',
         'R2_T1', 'R2_T2', 'R2_T3',
         'R3_T1', 'R3_T2', 'R3_T3']
df_best_parameters = pd.read_pickle(path_parameters)

vetAcc = []
vetPrec = []
vetF1 = []
vetRec = []
vetAverages = []

for k_bits in range(start_k, max_k + 1, 1):

    av_acc_test = 0
    av_prec_test = 0
    av_f1_test = 0
    av_rec_test = 0
    print('\n*************************************')
    print('k = ', k_bits, 'bit before and after')

    for i in range(0, len(loads), 1):

        par1_test = loads[i]
        par2_test = loads[i]

        dataTrainVal = split_TRAIN.TRAIN(df_in, par1_test, par2_test)
        dataset_validation, dataset_train = split_VALIDATION.setVal(dataTrainVal, val_size=0.01)
        dataset_test = split_TEST.TEST(df_in, par1_test, par2_test)

        final_col = len(df_in.columns) - 1

        X_val = dataset_validation.values[:, 1:final_col - 1]
        y_val = dataset_validation['D_class']

        X_train = dataset_train.values[:, 1:final_col - 1]
        y_train = dataset_train['D_class']

        X_test = dataset_test.values[:, 1:final_col - 1]
        y_test = dataset_test['D_class']

        best_parameters = get_best_parameters(df_best_parameters, par1_test, par2_test)

        clf = tree.DecisionTreeClassifier(**best_parameters, random_state=0)

        clf.fit(X_train, y_train)

        p_train = clf.predict(X_train)
        p_test = clf.predict(X_test)
        previous_acc_test = metrics.accuracy_score(y_test, p_test)

        p_test = evaluate_undamaged(p_test, k_bits)
        p_test = evaluate_damaged(p_test, k_bits)

        acc_train = metrics.accuracy_score(y_train, p_train)

        acc_test = metrics.accuracy_score(y_test, p_test)
        rec = metrics.recall_score(y_test, p_test)
        f1 = metrics.f1_score(y_test, p_test)
        prec = metrics.precision_score(y_test, p_test)

        print(f'{loads[i]}) New test accuracy: {round(acc_test * 100, 2)} % - Past test accuracy: {round(previous_acc_test * 100, 0)} %')

        av_acc_test = av_acc_test + acc_test
        av_f1_test = av_f1_test + f1
        av_prec_test = av_prec_test + prec
        av_rec_test = av_rec_test + rec

    average_averages = round((round(av_acc_test / len(loads) * 100, 2) + round(av_rec_test / len(loads) * 100, 2)
                              + round(av_f1_test / len(loads) * 100, 2) + round(av_prec_test / len(loads) * 100,
                                                                                2)) / 4, 2)

    vetAcc.append(round(av_acc_test / len(loads) * 100, 2))
    vetRec.append(round(av_rec_test / len(loads) * 100, 2))
    vetPrec.append(round(av_prec_test / len(loads) * 100, 2))
    vetF1.append(round(av_f1_test / len(loads) * 100, 2))
    vetAverages.append(average_averages)

    print(f'\nAverage test accuracy: {round(av_acc_test / len(loads) * 100, 2)} % ')
    print(f'Average test recall: {round(av_rec_test / len(loads) * 100, 2)} % ')
    print(f'Average test f1-score: {round(av_f1_test / len(loads) * 100, 2)} % ')
    print(f'Average test precision: {round(av_prec_test / len(loads) * 100, 2)} % ')

    print(f'\nAverage averages: {average_averages} % ')

x_axis = np.arange(start_k, max_k+1, 1)
fig = plt.figure(dpi=500)
plt.plot(x_axis, vetAcc, linewidth=3)
plt.plot(x_axis, vetRec, linewidth=3)
plt.plot(x_axis, vetF1, linewidth=3)
plt.plot(x_axis, vetPrec, linewidth=3)
plt.plot(x_axis, vetAverages, linewidth=5)
plt.xlabel('K bits', fontsize=fontsize)
plt.ylabel('Metrics [%]', fontsize=fontsize)
plt.xticks(np.arange(start_k, max_k+1, 1), fontsize=label_size_plot)
plt.yticks(fontsize=label_size_plot)
plt.legend(['Accuracy', 'Recall', 'F1-score', 'Precision', 'Averages'], fontsize=13)
plt.grid()
plt.savefig('KBitsFilter.png')
plt.show()
