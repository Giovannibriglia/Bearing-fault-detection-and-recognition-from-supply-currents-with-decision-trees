import split_TEST
import split_TRAIN
import split_VALIDATION
import pandas as pd
from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt

label_size_plot = 12
fontsize = 12
start_k, max_k, step_k = 0, 20, 1

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


def neighbors_based_filter(predictions, k):

    for i in range(len(predictions)):

        if i == 0:
            k_before = predictions[i]
        else:
            if i - k < 0:
                low_index = 0
            else:
                low_index = i - k

            if low_index == i:
                k_before = predictions[i]
            else:
                k_before = np.mean(predictions[low_index: i])

        if i == len(predictions)-1:
            k_after = predictions[len(predictions)-1]
        else:
            if i + k > len(predictions)-1:
                high_index = len(predictions)-1
            else:
                high_index = i + k

            if high_index == i:
                k_after = predictions[i]
            else:
                k_after = np.mean(predictions[i: high_index])

        if (k_before+k_after)/2 > 0.5:
            predictions[i] = 1
        elif (k_before+k_after)/2 < 0.5:
            predictions[i] = 0
        # if equal to 0.5: no changes

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

for k_bits in range(start_k, max_k + 1, step_k):

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
        previous_acc_test = metrics.accuracy_score(y_test, p_test)

        new_p_undamaged = neighbors_based_filter(p_test[0:int(len(p_test) / 3)], k_bits)
        new_p_damaged = neighbors_based_filter(p_test[int(len(p_test) / 3):len(p_test)], k_bits)

        p_test_new = []
        p_test_new.extend(new_p_undamaged)
        p_test_new.extend(new_p_damaged)

        acc_test = metrics.accuracy_score(y_test, p_test_new)
        rec = metrics.recall_score(y_test, p_test_new)
        f1 = metrics.f1_score(y_test, p_test_new)
        prec = metrics.precision_score(y_test, p_test_new)

        print(f'{loads[i]}) New test accuracy: {round(acc_test * 100, 2)} % - Past test accuracy: {round(previous_acc_test * 100, 2)} %')

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

x_axis = np.arange(start_k, max_k+1, step_k)
fig = plt.figure(dpi=500)
fig.suptitle('Neighbors-based filter', fontsize=fontsize+2)
plt.plot(x_axis, vetAcc, linewidth=3)
plt.plot(x_axis, vetRec, linewidth=3)
plt.plot(x_axis, vetF1, linewidth=3)
plt.plot(x_axis, vetPrec, linewidth=3)
plt.plot(x_axis, vetAverages, linewidth=5)
plt.xlabel('K bits', fontsize=fontsize)
plt.ylabel('Metrics [%]', fontsize=fontsize)
plt.xticks(x_axis, fontsize=label_size_plot)
plt.yticks(fontsize=label_size_plot)
plt.legend(['Accuracy', 'Recall', 'F1-score', 'Precision', 'Averages'], fontsize=13)
plt.grid()
plt.savefig('Neighbors_based_filter.png')
plt.show()